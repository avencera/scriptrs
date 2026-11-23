use crate::decode::{RawToken, RawTranscription};

/// Most encoder frames apart two tokens can be and still count as the same token
const MAX_MATCH_FRAMES: usize = 25;

/// Merges consecutive overlapping windows into one transcription whose token
/// frames never decrease
pub(crate) fn merge_overlapping_windows(windows: Vec<RawTranscription>) -> RawTranscription {
    let mut iter = windows.into_iter();
    let Some(mut merged) = iter.next() else {
        return RawTranscription::empty();
    };

    for window in iter {
        merged = merge_pair(merged, window);
    }
    merged
}

fn merge_pair(left: RawTranscription, right: RawTranscription) -> RawTranscription {
    if left.is_empty() {
        return right;
    }
    if right.is_empty() {
        return left;
    }

    // `left` is everything merged so far; only its tail can reach `right`, so the
    // search starts there rather than rescanning the whole transcript per window
    let left_start = first_reachable(&left, &right);
    let seam = match longest_agreement(&left, left_start, &right) {
        Some(run) => run.seam(&left, &right),
        None => Seam::at_midpoint(&left, &right),
    };
    seam.join(left, right)
}

/// Index of the first `left` token within match distance of any `right` frame; no
/// earlier token can match, so skipping them changes no match
fn first_reachable(left: &RawTranscription, right: &RawTranscription) -> usize {
    let Some(right_first) = right.tokens.iter().map(|token| token.frame).min() else {
        return left.tokens.len();
    };
    left.tokens
        .iter()
        .position(|token| token.frame + MAX_MATCH_FRAMES >= right_first)
        .unwrap_or(left.tokens.len())
}

/// A run of consecutive tokens both windows transcribed the same, token for token
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AgreedRun {
    left: usize,
    right: usize,
    len: usize,
}

impl AgreedRun {
    /// Where to hand over from `left` to `right` inside this run
    ///
    /// Both windows hold the same token ids along the run, so any handover point
    /// gives the same text and only decides whose timestamps the run's tokens keep.
    /// The handover point is the run's latest one whose join keeps frames in order.
    /// The run's end is not always such a point: tokens up to `MAX_MATCH_FRAMES`
    /// apart still match, so when one window drops a short restart the other kept,
    /// such as "solve a similar" against "solve a this a similar", the run's tail
    /// pairs left's word with right's earlier copy of it, and right's next token
    /// then starts before left's last
    fn seam(self, left: &RawTranscription, right: &RawTranscription) -> Seam {
        (0..=self.len)
            .rev()
            .map(|offset| Seam {
                left_end: self.left + offset,
                right_start: self.right + offset,
            })
            .find(|seam| seam.is_ordered(left, right))
            // no handover point keeps frames in order, as when right's timing runs
            // well ahead of left's along the whole run; the run's end keeps the
            // most of `left`, and `join` drops the right tokens that run back
            .unwrap_or(Seam {
                left_end: self.left + self.len,
                right_start: self.right + self.len,
            })
    }
}

/// The longest run of matching consecutive tokens, the earliest one on a tie
fn longest_agreement(
    left: &RawTranscription,
    left_start: usize,
    right: &RawTranscription,
) -> Option<AgreedRun> {
    let mut best: Option<AgreedRun> = None;
    for left_idx in left_start..left.tokens.len() {
        for right_idx in 0..right.tokens.len() {
            let len = (0..)
                .take_while(|offset| {
                    left_idx + offset < left.tokens.len()
                        && right_idx + offset < right.tokens.len()
                        && tokens_match(left, left_idx + offset, right, right_idx + offset)
                })
                .count();
            if len > best.map_or(0, |run| run.len) {
                best = Some(AgreedRun {
                    left: left_idx,
                    right: right_idx,
                    len,
                });
            }
        }
    }
    best
}

fn tokens_match(
    left: &RawTranscription,
    left_idx: usize,
    right: &RawTranscription,
    right_idx: usize,
) -> bool {
    let (left, right) = (left.tokens[left_idx], right.tokens[right_idx]);
    left.id == right.id && left.frame.abs_diff(right.frame) <= MAX_MATCH_FRAMES
}

/// Where a merge hands over: `left` up to `left_end`, then `right` from
/// `right_start`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Seam {
    left_end: usize,
    right_start: usize,
}

impl Seam {
    /// With no token in common, each window keeps the tokens on its side of the
    /// midpoint between left's last frame and right's first
    fn at_midpoint(left: &RawTranscription, right: &RawTranscription) -> Self {
        let left_cutoff = left.tokens.last().map_or(0, |token| token.frame);
        let right_first = right
            .tokens
            .first()
            .map_or(left_cutoff, |token| token.frame);
        let midpoint = (left_cutoff + right_first) / 2;

        Self {
            left_end: left.tokens.partition_point(|token| token.frame < midpoint),
            right_start: right.tokens.partition_point(|token| token.frame < midpoint),
        }
    }

    /// Whether right's first kept token starts no earlier than left's last
    fn is_ordered(self, left: &RawTranscription, right: &RawTranscription) -> bool {
        let Some(last_left) = self.left_end.checked_sub(1) else {
            return true;
        };
        right
            .tokens
            .get(self.right_start)
            .is_none_or(|first_right| first_right.frame >= left.tokens[last_left].frame)
    }

    /// Joins the two sides, dropping any `right` token that would start before
    /// the token kept ahead of it
    ///
    /// Each window's frames already rise, so with an ordered seam this drops
    /// nothing; it holds the merged order whatever the windows hold, since a
    /// token that runs back is a second copy of audio already transcribed
    fn join(self, left: RawTranscription, right: RawTranscription) -> RawTranscription {
        let mut tokens = left.tokens;
        tokens.truncate(self.left_end);
        let mut last_frame = tokens.last().map_or(0, |token| token.frame);
        let rest = right.tokens.into_iter().skip(self.right_start);
        tokens.extend(rest.filter(|token: &RawToken| {
            let in_order = token.frame >= last_frame;
            last_frame = last_frame.max(token.frame);
            in_order
        }));
        RawTranscription { tokens }
    }
}

#[cfg(test)]
mod tests {
    use super::merge_overlapping_windows;
    use crate::decode::RawTranscription;
    use crate::decode::test_support::raw_from_columns;

    fn window(tokens: &[(u32, usize)]) -> RawTranscription {
        let ids: Vec<u32> = tokens.iter().map(|(id, _)| *id).collect();
        let frames: Vec<usize> = tokens.iter().map(|(_, frame)| *frame).collect();
        raw_from_columns(&ids, &frames, &vec![1; ids.len()], &vec![0.9; ids.len()])
    }

    fn assert_frames_rise(merged: &RawTranscription) {
        let frames = merged.frames();
        assert!(
            frames.windows(2).all(|pair| pair[0] <= pair[1]),
            "frames run backward: {frames:?}"
        );
    }

    #[test]
    fn contiguous_tokens_are_deduplicated() {
        let left = raw_from_columns(&[1, 2, 3], &[0, 10, 20], &[1, 1, 1], &[0.9; 3]);
        let right = raw_from_columns(&[2, 3, 4], &[10, 20, 30], &[1, 1, 1], &[0.9; 3]);
        let merged = merge_overlapping_windows(vec![left, right]);
        assert_eq!(merged.ids(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn midpoint_fallback_keeps_monotonic_frames() {
        let left = raw_from_columns(&[1, 2], &[0, 10], &[1, 1], &[0.9; 2]);
        let right = raw_from_columns(&[9, 10], &[8, 18], &[1, 1], &[0.9; 2]);
        let merged = merge_overlapping_windows(vec![left, right]);
        assert_frames_rise(&merged);
    }

    /// Two windows of "To solve a this a similar": the earlier window heard only
    /// "To solve a similar", so the matched run ends on its "a" at frame 615,
    /// paired with the later window's first "a" at 607
    ///
    /// Frames are from a Parakeet v3 run on a podcast episode, where the run's end
    /// put "this" at 611 after "a" at 615
    #[test]
    fn dropped_restart_keeps_frames_in_order() {
        const TO: u32 = 1;
        const SOL: u32 = 2;
        const VE: u32 = 3;
        const A: u32 = 4;
        const THIS: u32 = 5;
        const SIM: u32 = 6;
        const NEED: u32 = 7;
        let left = window(&[(TO, 598), (SOL, 602), (VE, 605), (A, 615), (SIM, 616)]);
        let right = window(&[
            (TO, 598),
            (SOL, 602),
            (VE, 605),
            (A, 607),
            (THIS, 611),
            (A, 615),
            (SIM, 616),
            (NEED, 620),
        ]);

        let merged = merge_overlapping_windows(vec![left, right]);

        assert_eq!(merged.ids(), vec![TO, SOL, VE, A, THIS, A, SIM, NEED]);
        assert_eq!(
            merged.frames(),
            vec![598, 602, 605, 607, 611, 615, 616, 620]
        );
    }

    /// "But I say" in the earlier window against "But I think but I say" in the
    /// later one, which matches "But I" 13 frames early
    #[test]
    fn restart_heard_late_in_the_run_takes_the_later_windows_frames() {
        const BIG: u32 = 1;
        const STOP: u32 = 2;
        const BUT_UPPER: u32 = 3;
        const I: u32 = 4;
        const THINK: u32 = 5;
        const BUT: u32 = 6;
        const SAY: u32 = 7;
        let left = window(&[
            (BIG, 48931),
            (STOP, 48934),
            (BUT_UPPER, 48949),
            (I, 48950),
            (SAY, 48951),
        ]);
        let right = window(&[
            (BIG, 48931),
            (STOP, 48932),
            (BUT_UPPER, 48936),
            (I, 48937),
            (THINK, 48938),
            (BUT, 48949),
            (I, 48951),
            (SAY, 48952),
        ]);

        let merged = merge_overlapping_windows(vec![left, right]);

        assert_eq!(
            merged.ids(),
            vec![BIG, STOP, BUT_UPPER, I, THINK, BUT, I, SAY]
        );
        assert_eq!(
            merged.frames(),
            vec![48931, 48934, 48936, 48937, 48938, 48949, 48951, 48952]
        );
    }

    /// A run whose every handover point runs back still merges in order, by
    /// dropping the later window's tokens that start before the kept ones
    #[test]
    fn unordered_run_drops_tokens_that_run_back() {
        let left = window(&[(9, 98), (1, 100), (2, 120)]);
        let right = window(&[(1, 90), (2, 95), (3, 110), (4, 130)]);

        let merged = merge_overlapping_windows(vec![left, right]);

        assert_eq!(merged.ids(), vec![9, 1, 2, 4]);
        assert_eq!(merged.frames(), vec![98, 100, 120, 130]);
    }

    /// Merges many windows of a small vocabulary, where repeats and near-miss
    /// timings are common, and checks that frames never run backward
    #[test]
    fn merged_frames_never_run_backward() {
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };

        for _ in 0..2_000 {
            let mut windows = Vec::new();
            let mut start = 0usize;
            for _ in 0..4 {
                let mut frame = start;
                let tokens: Vec<(u32, usize)> = (0..30)
                    .map(|_| {
                        frame += next(4) as usize;
                        (next(4) as u32, frame)
                    })
                    .collect();
                windows.push(window(&tokens));
                start += 20 + next(30) as usize;
            }

            assert_frames_rise(&merge_overlapping_windows(windows));
        }
    }
}
