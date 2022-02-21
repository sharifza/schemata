## [Classification by Attention: Scene Graph Classification with Prior Knowledge (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/16636)

#### by [Sahand Sharifzadeh](https://www.linkedin.com/in/sahandsharifzadeh/ )<sup>1</sup>, [Sina Moayed Baharlou](https://www.sinabaharlou.com)<sup>1</sup>, [Volker Tresp](https://www.dbs.ifi.lmu.de/~tresp/ )<sup>1,2</sup>
<sup>1 </sup> Ludwig Maximilian University, Munich, Germany,
<sup>2 </sup> Siemens AG, Munich, Germany<br/>

## Abstract
A major challenge in scene graph classification is that the appearance of objects and relations can be significantly different from one image to another. 
Previous works have addressed this by relational reasoning over all objects in an image or incorporating prior knowledge into classification. 
Unlike previous works, we do not consider separate models for perception and prior knowledge.
Instead, we take a multi-task learning approach, where we implement the classification as an attention layer. This allows for the prior knowledge to emerge and propagate within the perception model. By enforcing the model also to represent the prior, we achieve a strong inductive bias. 
We show that our model can accurately generate commonsense knowledge and that the iterative injection of this knowledge to scene representations leads to significantly higher classification performance. Additionally, our model can be fine-tuned on external knowledge given as triples. 
When combined with self-supervised learning and with 1% of annotated images only, this gives more than 3% improvement in object classification, 26% in scene graph classification, and 36% in predicate prediction accuracy.

## Model
<p align="center"><img src="docs/schema-architecture.png" title="Schema architecture."></p>

## Results
<p align="center"><img src="docs/R@K.png" title="R@K Results"></p>
<p align="center"><img src="docs/mR@K.png" title="mR@K Results"></p>

## Splits of Visual Genome

You can find the proposed splits of the VG dataset in `schemata/splits` (1% and 10% with 4 splits each).

## Notes
* The skeleton of our code is built on top of the nice framework of [Neural-Motifs](https://github.com/rowanz/neural-motifs). This includes the data loading pipeline, and part of the evaluation code. 
However, we have updated these parts to be compatible with PyTorch >= 1.

* The code is tested under Scene Graph Classification (SGCls) and Predicate Prediction (PredCls) settings.
We had difficulty reproducing Scene Graph _Detection_ (SGDet) of Neural Motifs under the updated PyTorch, and 
nevertheless, SGDet was not the focus of our experiments. 

* This code also contains our implementations of mR@K, which was originally not available in the NM code.

* The current release only presents the results for assimilation and not the accomodation. However, the IZS splits are
available.


## Bibtex
```
@article{Sharifzadeh_Moayed Baharlou_Tresp_2021, 
  title={Classification by Attention: Scene Graph Classification with Prior Knowledge}, 
  volume={35}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/16636}, 
  number={6}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Sharifzadeh, Sahand and Moayed Baharlou, Sina and Tresp, Volker}, 
  year={2021}, 
  month={May}, 
  pages={5025-5033} 
}
```