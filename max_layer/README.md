## Processed data link
- https://drive.google.com/file/d/1OJLAS8jkElmGEGVIa8wz-Pg2BRCbqE2q/view?usp=drive_link
 
## Example of processed data with max layer info
```json
[
    {
        "low_score": 0.01815050281584263,
        "high_score": 0.0948466807603836,
        "input_ids": [
            53,
            7899,
            5674,
            361,
            318,
            5140,
            287,
            262,
            15549,
            286
        ],
        "input_tokens": [
            "V",
            "inson",
            " Mass",
            "if",
            " is",
            " located",
            " in",
            " the",
            " continent",
            " of"
        ],
        "subject_range": [
            0,
            4
        ],
        "answer": " Antarctica",
        "window": 10,
        "correct_prediction": true,
        "kind": "mlp",
        "prompt": "Vinson Massif is located in the continent of",
        "subject": "Vinson Massif",
        "known_id": 0,
        "relation_id": "P30",
        "max_score_layer": [
            27
        ]
    },
    {
        "low_score": 0.004500404931604862,
        "high_score": 0.683708906173706,
        "input_ids": [
            3856,
            1381,
            7849,
            318,
            6898,
            416
        ],
        "input_tokens": [
            "Be",
            "ats",
            " Music",
            " is",
            " owned",
            " by"
        ],
        "subject_range": [
            0,
            3
        ],
        "answer": " Apple",
        "window": 10,
        "correct_prediction": true,
        "kind": "mlp",
        "prompt": "Beats Music is owned by",
        "subject": "Beats Music",
        "known_id": 1,
        "relation_id": "P127",
        "max_score_layer": [
            9
        ]
    }
]
```