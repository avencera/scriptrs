use std::ops::Range;

use crate::decode::RawTranscription;

/// Most encoder frames apart two tokens can be and still count as the same token
const MAX_MATCH_FRAMES: usize = 25;

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
    // searches start there rather than rescanning the whole transcript per window
    let left_start = first_reachable(&left, &right);
    let matches = contiguous_matches(&left, left_start, &right);
    if !matches.is_empty() {
        return merge_using_matches(left, right, &matches);
    }

    let matches = lcs_matches(&left, left_start, &right);
    if !matches.is_empty() {
        return merge_using_matches(left, right, &matches);
    }

    merge_by_midpoint(left, right)
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

fn contiguous_matches(
    left: &RawTranscription,
    left_start: usize,
    right: &RawTranscription,
) -> Vec<(usize, usize)> {
    let mut best = Vec::new();
    for left_idx in left_start..left.tokens.len() {
        for right_idx in 0..right.tokens.len() {
            if !tokens_match(left, left_idx, right, right_idx) {
                continue;
            }
            let mut current = Vec::new();
            let mut lhs = left_idx;
            let mut rhs = right_idx;
            while lhs < left.tokens.len() && rhs < right.tokens.len() {
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

/// Longest common subsequence of `left[left_start..]` and `right`; rows before
/// `left_start` would all be zero, so the table and its traceback start there
fn lcs_matches(
    left: &RawTranscription,
    left_start: usize,
    right: &RawTranscription,
) -> Vec<(usize, usize)> {
    let left_len = left.tokens.len() - left_start;
    let matches_at = |row: usize, col: usize| tokens_match(left, left_start + row, right, col);
    let mut dp = vec![vec![0usize; right.tokens.len() + 1]; left_len + 1];
    for left_idx in 1..=left_len {
        for right_idx in 1..=right.tokens.len() {
            if matches_at(left_idx - 1, right_idx - 1) {
                dp[left_idx][right_idx] = dp[left_idx - 1][right_idx - 1] + 1;
            } else {
                dp[left_idx][right_idx] =
                    dp[left_idx - 1][right_idx].max(dp[left_idx][right_idx - 1]);
            }
        }
    }

    let mut matches = Vec::new();
    let mut left_idx = left_len;
    let mut right_idx = right.tokens.len();
    while left_idx > 0 && right_idx > 0 {
        if matches_at(left_idx - 1, right_idx - 1) {
            matches.push((left_start + left_idx - 1, right_idx - 1));
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
    let (left, right) = (left.tokens[left_idx], right.tokens[right_idx]);
    left.id == right.id && left.frame.abs_diff(right.frame) <= MAX_MATCH_FRAMES
}

fn merge_using_matches(
    left: RawTranscription,
    right: RawTranscription,
    matches: &[(usize, usize)],
) -> RawTranscription {
    let left_indices: Vec<usize> = matches.iter().map(|(left_idx, _)| *left_idx).collect();
    let right_indices: Vec<usize> = matches.iter().map(|(_, right_idx)| *right_idx).collect();

    let mut merged = RawTranscription::empty();

    if let Some(first_left) = left_indices.first().copied() {
        copy_tokens(&mut merged, &left, 0..first_left);
    }

    for (match_idx, (left_idx, right_idx)) in matches.iter().copied().enumerate() {
        copy_tokens(&mut merged, &left, left_idx..left_idx + 1);
        if match_idx == matches.len() - 1 {
            continue;
        }
        let next_left = left_indices[match_idx + 1];
        let next_right = right_indices[match_idx + 1];
        let left_gap = next_left.saturating_sub(left_idx + 1);
        let right_gap = next_right.saturating_sub(right_idx + 1);
        if right_gap > left_gap {
            copy_tokens(&mut merged, &right, right_idx + 1..next_right);
        } else {
            copy_tokens(&mut merged, &left, left_idx + 1..next_left);
        }
    }

    if let Some(last_right) = right_indices.last().copied() {
        copy_tokens(&mut merged, &right, last_right + 1..right.tokens.len());
    }

    merged
}

fn merge_by_midpoint(left: RawTranscription, right: RawTranscription) -> RawTranscription {
    let left_cutoff = left.tokens.last().map_or(0, |token| token.frame);
    let right_start = right
        .tokens
        .first()
        .map_or(left_cutoff, |token| token.frame);
    let midpoint = (left_cutoff + right_start) / 2;

    let mut merged = RawTranscription::empty();
    for token in &left.tokens {
        if token.frame < midpoint {
            merged.tokens.push(*token);
        }
    }
    for token in &right.tokens {
        if token.frame >= midpoint {
            merged.tokens.push(*token);
        }
    }
    merged
}

/// Appends `source`'s tokens at `range`, which may be empty or reversed
fn copy_tokens(merged: &mut RawTranscription, source: &RawTranscription, range: Range<usize>) {
    merged
        .tokens
        .extend(range.map(|index| source.tokens[index]));
}

#[cfg(test)]
mod tests {
    use super::merge_overlapping_windows;
    use crate::decode::test_support::raw_from_columns;

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
        assert!(
            merged
                .frames()
                .windows(2)
                .all(|window| window[0] <= window[1])
        );
    }
}
