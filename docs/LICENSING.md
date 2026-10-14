# Licensing notes for release preparation

Lev source code is Apache-2.0, derived from Kev. Its LICENSE is copied verbatim, and NOTICE identifies the upstream project and changes. Weight and dataset terms are separate.

The fixed-revision model cards for ByteDance/Ouro-1.4B (`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`) and ByteDance/Ouro-2.6B (`1ed04250da1a9936042725d302e81c8fa2ab5abd`) state `license: apache-2.0` and say in their license section that the model is licensed under Apache-2.0. They also contain: “This model is intended for research purposes only. It is provided as-is without warranties for production use.” This is reproduced as the upstream intended-use statement, not converted into a new license restriction by this repository. No separate LICENSE/NOTICE file was present in those fixed-revision Hub file listings when checked on 2026-10-13. Confirm release obligations against the complete upstream materials before weight publication.

Nanbeige/Nanbeige4.2-3B-Base (`4a38e817c14b9f2c4b69b1e05cbda90395872f6b`) declares Apache-2.0 in its fixed-revision model-card metadata. No separate LICENSE/NOTICE file was listed there in that check. The Lev model cards remain publication drafts until adapter/head artifacts, license notices and final Hub revisions are verified.

Dataset statements in `data/sources.json` are provenance records, not newly granted licenses. They distinguish recorded terms from release approval. This source candidate distributes no dataset text, and does not relicense the aggregated external benchmark results or third-party material through the code license.
