# TransformerPivotTranslation
<h1>
Neural Machine Translation for Polish–German  
via Czech as a Pivot Language
</h1>

<div class="section">
<h2>1. Authors</h2>

<p>
Naohiro Tokunaga (Ehime University)  
<br>
Takashi Ninomiya (Ehime University)  
<br>
Akihiro Tamura (Doshisha University)
</p>

<p>
Presented at the Shikoku-Section Joint Conference.
</p>
</div>

<div class="section">
<h2>2. Research Background</h2>

<p>
When sufficient parallel corpora exist for language pairs A–C and C–B,
but not for A–B directly, pivot translation becomes an effective strategy.
In pivot translation, an intermediate language C is used to bridge
the source language A and the target language B.
</p>

<p>
This study focuses on Polish–German translation,
where Czech is employed as the default pivot language.
Because Polish and Czech both belong to the West Slavic language group,
we hypothesize that Czech may serve as a more suitable pivot language
than English.
</p>

<p>
For comparison, we also conduct experiments using English as the pivot language.
</p>
</div>

<div class="section">
<h2>3. Proposed Method</h2>

<p>
We construct two Transformer-based neural machine translation models:
</p>

<ol>
<li>Polish → Czech</li>
<li>Czech → German</li>
</ol>

<p>
Translation is performed sequentially through these two models.
</p>

<p>
The Transformer architecture follows the standard encoder–decoder structure
based on multi-head self-attention and position-wise feed-forward networks.
</p>
</div>

<div class="section">
<h2>4. Experimental Setup</h2>

<h3>4.1 Corpus</h3>
<p>
We use the JRC-Acquis parallel corpus.
</p>

<h3>4.2 Data Size</h3>
<ul>
<li>Training data: 60,000 sentence pairs per translation model</li>
<li>Test data: 10,000 sentences</li>
</ul>

<h3>4.3 Data Split</h3>
<p>
The corpus is divided into training, validation, and test sets
in the ratio 9 : 0.5 : 0.5.
</p>

<h3>4.4 Evaluation Metrics</h3>
<ul>
<li>BLEU</li>
<li>GLEU</li>
<li>CHRF</li>
</ul>
</div>

<div class="section">
<h2>5. Results</h2>

<h3>5.1 Czech as Pivot Language</h3>

<table>
<tr>
<th>Metric</th>
<th>Average Score</th>
</tr>
<tr>
<td>BLEU</td>
<td>0.0203</td>
</tr>
<tr>
<td>GLEU</td>
<td>0.0492</td>
</tr>
<tr>
<td>CHRF</td>
<td>0.1870</td>
</tr>
</table>

<h3>5.2 English as Pivot Language</h3>

<table>
<tr>
<th>Metric</th>
<th>Average Score</th>
</tr>
<tr>
<td>BLEU</td>
<td>0.0188</td>
</tr>
<tr>
<td>GLEU</td>
<td>0.0281</td>
</tr>
<tr>
<td>CHRF</td>
<td>0.1465</td>
</tr>
</table>

<p>
The results indicate that Czech-based pivot translation
outperforms English-based pivot translation across all evaluation metrics.
</p>
</div>

<div class="section">
<h2>6. Dataset Availability and Citation</h2>

<p>
For the purpose of reproducibility and transparency,
the processed corpus used in this study is publicly available via
Hugging Face Datasets at the following URL:
</p>

<p>
<a href="https://huggingface.co/datasets/SHSK0118/TransformerPivotTranslation">
https://huggingface.co/datasets/SHSK0118/TransformerPivotTranslation
</a>
</p>

<p>
The dataset includes the preprocessed training, validation,
and test splits used in our experiments.
</p>

<h3>6.1 Recommended Citation</h3>

<p>
If you use this dataset or implementation in your research,
please cite it as follows:
</p>

<div class="codeblock">
Tokunaga, N., Ninomiya, T., & Tamura, A. (Year).
TransformerPivotTranslation Dataset.
Hugging Face. 
https://huggingface.co/datasets/SHSK0118/TransformerPivotTranslation
</div>

<p>
Please replace <em>Year</em> with the publication year.
</p>

</div>

<div class="section">
<h2>7. Future Work</h2>

<ul>
<li>Use of larger parallel corpora</li>
<li>Introduction of subword segmentation methods (e.g., BPE)</li>
<li>Comparison with direct Polish–German translation</li>
<li>Incorporation of pre-trained multilingual models</li>
</ul>
</div>

<div class="section">
<h2>9. Directory Structure</h2>

<div class="codeblock">
TransformerPivotTranslation
├ requirements.txt
├ README.md
└ src
│ ├ model.py
│ ├ preprocessing.py
│ ├ train.py
│ ├ test.py
│ └ saved_model
└ corpus
  ├ JRC-Acquis
  └ split
</div>

</div>
