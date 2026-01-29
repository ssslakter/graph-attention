#set page(
  margin: (x: 2.5cm, y: 2.5cm),
)
#show ref: it => {
  let el = it.element
  if el != none and el.func() == heading {
    link(el.location(), el.body)
  } else {
    it
  }
}

#set text(
  font: "New Computer Modern",
  size: 12pt,
)

#show heading: set text(weight: "bold")
#show heading.where(level: 1): it => {
  v(1em)
  it
  v(0.5em)
}

#text(
  font: "Futura Cyrillic",
  weight: "bold",
  size: 44pt,
)[Skoltech]
#v(-3em)
ISP PROJECT REPORT

#text(weight: "bold", size: 12pt)[Graph Attention]


#v(0.5em)

Master's Educational Program: Data Science

#v(1em)

#align(center)[
  #block(width: 70%)[
    #line(length: 100%, stroke: 1pt)
    #grid(
      columns: (1fr, 1fr),
      gutter: 1em,
      align: (right, left),
      [Student:], [Viacheslav Chaunin],
      [Project:], [Graph Attention],
      [Supervisor:], [Anh Huy Phan],
    )
    #line(length: 100%, stroke: 1pt)
  ]
]

#show outline.entry.where(level: 1): it => {
  v(0.5em)
  it
}

#outline(
  depth: 1,
  title: [Contents],
  indent: auto,
)

= Project Purpose
The objective of this project is to investigate how the standard multi-head attention layer can be enhanced by reformulating it as a graph filtering problem. This approach aims to potentially accelerate model inference through the use of structured attention matrices or to improve model generalization capabilities with manageable performance overhead. Currently, the formulation is defined as follows:

$ X^((l+1)) = X^((l)) + sum^H_(h=1) sum^K_(k=1) alpha_(k,h) A^k_h V_h W_O $

where $A_h = "softmax"((Q_h K_h^T)/sqrt(d))$ with $Q_h=X W_(Q, h), K_h=X W_(K, h), "and" V_h=X W_(V, h)$.

Here, $A_h$ represents a learned, input-dependent adjacency matrix. Consequently, powers of this matrix ($A^k_h$) facilitate multi-hop interactions between tokens, allowing for richer structural information capture.

= Performed Tasks
The project began with the implementation of the proposed graph filtering layers within the Vision Transformer (ViT) architecture. Initial validation was conducted on the CIFAR-10 dataset to identify bugs and monitor training stability. 

To ensure stability for the learned coefficients ($alpha$), three parameterization variants were proposed:
- *Tanh:* Limits values to the interval $[-1, 1]$.
- *Sigmoid:* Ensures positive values.
- *Softmax:* Interprets coefficients as probabilities.

Refer to the @appendix for an analysis of the alpha contributions after training models using the `tanh` activation. Additionally, a zero-order term ($alpha_0 I$) was initially proposed to adjust the strength of the residual connection. However, as it demonstrated no negligible impact on performance, it was excluded from subsequent experiments.

Initial results on CIFAR-10 indicated that higher-order filtering improved accuracy. By incorporating data augmentation and extending the training duration, the baseline order-1 attention model achieved 92% accuracy on CIFAR-10. Higher-order models achieved comparable results. However, further reevaluation is required using different initialization and parameterization setups.

= Work Plan
The work plan is structured as follows:

1. *Performance Evaluation:* Assess the layer's performance within the ViT model on large-scale image datasets. While initial testing used CIFAR-10, the focus has shifted to ImageNet-1k.
2. *Hyperparameter Optimization:* Determine the optimal filter order and the most effective parameterization for the alpha coefficients.
3. *Framework Generalization:* Extend the graph filtering concept to create a unified framework for architectures like ResNet and ViT. 
   - *Current ResNet structure:* Convolution $->$ Activation $->$ Convolution.
   - *Current ViT structure:* Attention $->$ Down-projection $->$ Up-projection.
   - *Proposed Generalization:* Attention $->$ Down-projection (Conv) $->$ Activation $->$ Up-projection (Conv).
   
The immediate goal is to implement this generalized structure in a ResNet-style architecture, starting from adding standard attention mechanisms to the ResNet and convolutions to ViT baselines.

= Expected and Achieved Results
We anticipate that the new layer will yield improvements in classification accuracy for ViTs by enabling more complex information exchange between tokens. 

*Preliminary Results:* On the ImageNet-1k validation set, the model has currently achieved an accuracy of *74.61%*.


= Appendix <appendix>
The following figures illustrate the importance of attention heads across different layers and orders during CIFAR-10 training, using the $"tanh"(x)$ activation function.

#figure(
  image("2.png", width: 60%),
  caption: [
    Contributions of attention heads across layers for the *order-1* model.
  ],
)

#figure(
  image("3.png", width: 70%),
  caption: [
    Contributions of attention heads across layers for the *order-2* model.
  ],
)

#figure(
  image("6.png", width: 100%),
  caption: [
    Contributions of attention heads across layers for the *order-5* model.
  ],
)