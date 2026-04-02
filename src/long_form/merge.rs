use crate::decode::RawTranscription;

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

    let matches = contiguous_matches(&left, &right);
    if !matches.is_empty() {
        return merge_using_matches(left, right, &matches);
    }

    let matches = lcs_matches(&left, &right);
    if !matches.is_empty() {
        return merge_using_matches(left, right, &matches);
    }

    merge_by_midpoint(left, right)
}

fn contiguous_matches(left: &RawTranscription, right: &RawTranscription) -> Vec<(usize, usize)> {
    let mut best = Vec::new();
    for left_idx in 0..left.token_ids.len() {
        for right_idx in 0..right.token_ids.len() {
            if !tokens_match(left, left_idx, right, right_idx) {
                continue;
            }
            let mut current = Vec::new();
            let mut lhs = left_idx;
            let mut rhs = right_idx;
            while lhs < left.token_ids.len() && rhs < right.token_ids.len() {
                if !tokens_match(left, lhs, right, rhs) {
                    break;
                }
                current.push((lhs, rhs));
                lhs += 1;
                rhs += 1;
            }
            if current.len() > best.len() {
                best = current;
            }
        }
    }
    best
}

fn lcs_matches(left: &RawTranscription, right: &RawTranscription) -> Vec<(usize, usize)> {
    let mut dp = vec![vec![0usize; right.token_ids.len() + 1]; left.token_ids.len() + 1];
    for left_idx in 1..=left.token_ids.len() {
        for right_idx in 1..=right.token_ids.len() {
            if tokens_match(left, left_idx - 1, right, right_idx - 1) {
                dp[left_idx][right_idx] = dp[left_idx - 1][right_idx - 1] + 1;
            } else {
                dp[left_idx][right_idx] =
                    dp[left_idx - 1][right_idx].max(dp[left_idx][right_idx - 1]);
            }
        }
    }

    let mut matches = Vec::new();
    let mut left_idx = left.token_ids.len();
    let mut right_idx = right.token_ids.len();
    while left_idx > 0 && right_idx > 0 {
        if tokens_match(left, left_idx - 1, right, right_idx - 1) {
            matches.push((left_idx - 1, right_idx - 1));
            left_idx -= 1;
            right_idx -= 1;
            continue;
        }
        if dp[left_idx - 1][right_idx] > dp[left_idx][right_idx - 1] {
            left_idx -= 1;
        } else {
            right_idx -= 1;
        }
    }
    matches.reverse();
    matches
}

fn tokens_match(
    left: &RawTranscription,
    left_idx: usize,
    right: &RawTranscription,
    right_idx: usize,
) -> bool {
    left.token_ids[left_idx] == right.token_ids[right_idx]
        && left.frame_indices[left_idx].abs_diff(right.frame_indices[right_idx]) <= 25
}

fn merge_using_matches(
    left: RawTranscription,
    right: RawTranscription,
    matches: &[(usize, usize)],
) -> RawTranscription {
    let left_indices: Vec<usize> = matches.iter().map(|(left_idx, _)| *left_idx).collect();
    let right_indices: Vec<usize> = matches.iter().map(|(_, right_idx)| *right_idx).collect();

    let mut merged = RawTranscriptionBuilder::new();

    if let Some(first_left) = left_indices.first().copied() {
        merged.extend(&left, 0..first_left);
    }

    for (match_idx, (left_idx, right_idx)) in matches.iter().copied().enumerate() {
        merged.extend(&left, left_idx..left_idx + 1);
        if match_idx == matches.len() - 1 {
            continue;
        }
        let next_left = left_indices[match_idx + 1];
        let next_right = right_indices[match_idx + 1];
        let left_gap = next_left.saturating_sub(left_idx + 1);
        let right_gap = next_right.saturating_sub(right_idx + 1);
        if right_gap > left_gap {
            merged.extend(&right, right_idx + 1..next_right);
        } else {
            merged.extend(&left, left_idx + 1..next_left);
        }
    }

    if let Some(last_right) = right_indices.last().copied() {
        merged.extend(&right, last_right + 1..right.token_ids.len());
    }

    merged.finish()
}

fn merge_by_midpoint(left: RawTranscription, right: RawTranscription) -> RawTranscription {
    let left_cutoff = left.frame_indices.last().copied().unwrap_or(0);
    let right_start = right.frame_indices.first().copied().unwrap_or(left_cutoff);
    let midpoint = (left_cutoff + right_start) / 2;

    let mut merged = RawTranscriptionBuilder::new();

    for index in 0..left.token_ids.len() {
        if left.frame_indices[index] >= midpoint {
            continue;
        }
        merged.push(&left, index);
    }

    for index in 0..right.token_ids.len() {
        if right.frame_indices[index] < midpoint {
            continue;
        }
        merged.push(&right, index);
    }

    merged.finish()
}

#[derive(Debug, Default)]
struct RawTranscriptionBuilder {
    token_ids: Vec<u32>,
    frame_indices: Vec<usize>,
    durations: Vec<usize>,
    confidences: Vec<f32>,
}

impl RawTranscriptionBuilder {
    fn new() -> Self {
        Self::default()
    }

    fn extend(&mut self, source: &RawTranscription, range: std::ops::Range<usize>) {
        for index in range {
            self.push(source, index);
        }
    }

    fn push(&mut self, source: &RawTranscription, index: usize) {
        self.token_ids.push(source.token_ids[index]);
        self.frame_indices.push(source.frame_indices[index]);
        self.durations.push(source.durations[index]);
        self.confidences.push(source.confidences[index]);
    }

    fn finish(self) -> RawTranscription {
        RawTranscription {
            token_ids: self.token_ids,
            frame_indices: self.frame_indices,
            durations: self.durations,
            confidences: self.confidences,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::decode::RawTranscription;

    use super::merge_overlapping_windows;

    #[test]
    fn contiguous_tokens_are_deduplicated() {
        let left = RawTranscription {
            token_ids: vec![1, 2, 3],
            frame_indices: vec![0, 10, 20],
            durations: vec![1, 1, 1],
            confidences: vec![0.9; 3],
        };
        let right = RawTranscription {
            token_ids: vec![2, 3, 4],
            frame_indices: vec![10, 20, 30],
            durations: vec![1, 1, 1],
            confidences: vec![0.9; 3],
        };
        let merged = merge_overlapping_windows(vec![left, right]);
        assert_eq!(merged.token_ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn midpoint_fallback_keeps_monotonic_frames() {
        let left = RawTranscription {
            token_ids: vec![1, 2],
            frame_indices: vec![0, 10],
            durations: vec![1, 1],
            confidences: vec![0.9; 2],
        };
        let right = RawTranscription {
            token_ids: vec![9, 10],
            frame_indices: vec![8, 18],
            durations: vec![1, 1],
            confidences: vec![0.9; 2],
        };
        let merged = merge_overlapping_windows(vec![left, right]);
        assert!(
            merged
                .frame_indices
                .windows(2)
                .all(|window| window[0] <= window[1])
        );
    }
}
