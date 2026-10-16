//! Isolated segmentation timing on already normalized JSONL {kind, id, text} records.
#[path = "../src/word_segmentation.rs"]
mod reference;

use serde_json::{Value, json};
use std::{hint::black_box, io::BufRead, time::Instant};

fn main() {
    let input = std::env::args().nth(1).expect("normalized JSONL path");
    let rows: Vec<Value> = std::io::BufReader::new(std::fs::File::open(&input).unwrap())
        .lines()
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect();
    // Match the engine's non-query fragment boundaries; preparation is not timed.
    let fragments: Vec<Vec<String>> = rows
        .iter()
        .map(|row| {
            let text = row["text"].as_str().unwrap().to_lowercase();
            let mut start = None;
            let mut parts = Vec::new();
            for (offset, c) in text.char_indices() {
                if regex_syntax::is_word_character(c) {
                    start.get_or_insert(offset);
                } else if !matches!(c, '+' | '-' | '#') {
                    if let Some(begin) = start.take() {
                        parts.push(text[begin..offset].to_owned());
                    }
                }
            }
            if let Some(begin) = start {
                parts.push(text[begin..].to_owned());
            }
            parts
        })
        .collect();
    let mut reference = reference::WordSegmentationTM::new();
    reference.load_dictionary(0, 1, true);
    let jieba = jieba_rs::Jieba::new();
    let segment = |mode, text: &str| -> Vec<String> {
        if mode == "reference" {
            reference.segment(text, true).0
        } else if text.is_ascii() {
            vec![text.to_owned()]
        } else {
            jieba
                .cut(text, mode == "jieba")
                .into_iter()
                .map(|token| token.word.to_owned())
                .collect()
        }
    };
    let mut runs = Vec::new();
    for kind in ["document", "query"] {
        let group: Vec<_> = rows
            .iter()
            .zip(&fragments)
            .filter(|(row, _)| row["kind"] == kind)
            .map(|(_, parts)| parts)
            .collect();
        for repetition in 0..4 {
            let mut modes = ["reference", "jieba", "jieba_no_hmm"];
            modes.rotate_left(repetition % 3);
            for mode in modes {
                let start = Instant::now();
                let mut tokens = 0;
                for parts in &group {
                    for part in *parts {
                        tokens += black_box(segment(mode, black_box(part))).len();
                    }
                }
                let seconds = start.elapsed().as_secs_f64();
                if repetition != 0 {
                    runs.push(json!({"kind":kind,"mode":mode,"repetition":repetition,
                        "records":group.len(),"tokens":tokens,"seconds":seconds}));
                }
            }
        }
    }
    let queries: Vec<_> = rows
        .iter()
        .zip(&fragments)
        .filter(|(row, _)| row["kind"] == "query")
        .map(|(row, parts)| {
            json!({"id":row["id"],"text":row["text"],
            "reference":parts.iter().flat_map(|p| segment("reference",p)).collect::<Vec<_>>(),
            "jieba":parts.iter().flat_map(|p| segment("jieba",p)).collect::<Vec<_>>(),
            "jieba_no_hmm":parts.iter().flat_map(|p| segment("jieba_no_hmm",p)).collect::<Vec<_>>()})
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&json!({"input":input,
        "jieba_version":"0.11.0","hmm":{"jieba":true,"jieba_no_hmm":false},"runs":runs,"queries":queries,
        "method":"Same normalized fragments, ASCII bypass and owned token output; initialization, lowercasing and boundary splitting excluded. One warmup, three rotating-order measured repetitions. Not HTTP latency."})).unwrap());
}
