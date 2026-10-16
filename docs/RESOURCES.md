# Project resources

| Resource | Link | What it is |
|---|---|---|
| Project website | [openlair.github.io/lev](https://openlair.github.io/lev/) | Overview, interactive depth demo and results |
| Paper | [arXiv:2610.20951](https://arxiv.org/abs/2610.20951) · [PDF](https://arxiv.org/pdf/2610.20951) | Discussion on [alphaXiv](https://www.alphaxiv.org/abs/2610.20951) |
| Source code | [OpenLAIR/lev](https://github.com/OpenLAIR/lev) | Training, depth readouts, analysis and reproduction |
| Models and datasets | [Lev collection](https://huggingface.co/collections/songdj/lev) | Three models and three Dataset repositories |
| Lev-1.4B | [songdj/Lev-1.4B](https://huggingface.co/songdj/Lev-1.4B) | Ouro-1.4B backbone, depths 1–4, seed 0 |
| Lev-2.6B | [songdj/Lev-2.6B](https://huggingface.co/songdj/Lev-2.6B) | Ouro-2.6B backbone, depths 1–4, seed 0 |
| Lev-3B | [songdj/Lev-3B](https://huggingface.co/songdj/Lev-3B) | Nanbeige4.2-3B backbone, depths 1–2, seed 0 |
| LevBench | [songdj/LevBench](https://huggingface.co/datasets/songdj/LevBench) | Development and locked-test decisions across five tiers (rights-filtered subset) |
| Lev-train-v2 | [songdj/Lev-train-v2](https://huggingface.co/datasets/songdj/Lev-train-v2) | Training mixtures, replay and order recipes (rights-filtered subset) |
| Lev-results | [songdj/Lev-results](https://huggingface.co/datasets/songdj/Lev-results) | Per-depth outputs, calibration and latency, without question text |

The hosted dataset subsets do not constitute the complete frozen corpus: sources whose redistribution rights are unclear are left out and can be rebuilt locally from their original releases. See the [dataset cards](../data-cards/) and [data requirements](DATA.md). Source code, model artifacts and dataset text have separate licensing terms; see [licensing](LICENSING.md). What has and has not been re-run on the released packages is in [release status](RELEASE_STATUS.md).
