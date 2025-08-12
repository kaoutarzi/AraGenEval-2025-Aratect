# AraGenEval-2025-Aratect
# LMSA at AraGenEval 2025
## Overview
This repository contains the code and resources for our submission to the AraGenEval Shared Task 2025, where our system focuses on detecting AI-generated Arabic text.
Our approach combines multiple multilingual and Arabic-specific pre-trained language models and uses an ensemble voting mechanism for robust performance.
## Train Models
python fanar.py


## Results

| Model               | Acc.   | Prec.  | Rec.   | F1     |
|---------------------|--------|--------|--------|--------|
| LR                  | 0.438  | 0.464  | 0.804  | 0.589  |
| MLP                 | 0.506  | 0.503  | 0.988  | 0.667  |
| FusionNet           | 0.578  | 0.552  | 0.824  | 0.661  |
| AraElectra          | 0.688  | 0.737  | 0.584  | 0.652  |
| MARBERT             | 0.586  | 0.563  | 0.764  | 0.649  |
| DeBERTa             | 0.768  | 0.791  | 0.728  | 0.758  |
| Qwen2.5              | 0.480  | 0.490  | 0.940  | 0.644  |
| CAMeL               | 0.642  | 0.612  | 0.776  | 0.684  |
| XLM-R               | 0.832  | 0.911  | 0.736  | 0.814  |
| AraBERT             | 0.864  | 0.882  | 0.840  | 0.861  |
| Fanar               | 0.776  | 0.714  | 0.920  | 0.804  |
| **Majority Voting** | **0.866** | **0.877** | **0.852** | **0.864** |

##  BibTeX Citation
```bibtex
@inproceedings{Zita-anlp-2025,
  title = "{LMSA} at {AraGenEval} shared task: Ensemble-Based Detection of AI-Generated Arabic Text Using Multilingual and Arabic-Specific Models",
  author = "Zita, Kaoutar and Nehar, Attia and Khalil, Abdelkader and Bellaouar, Slimane and Cherroun, Hadda",
  booktitle = "Proceedings of the Third Arabic Natural Language Processing Conference",
  year = "2025",
  address = "Suzhou, China",
  publisher = "Association for Computational Linguistics (ACL)"
}
