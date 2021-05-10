# Metrics

In this part, we provide three metrics here as shown in the paper:

* Fidelity+
* Fidelity-
* Sparsity

## Quick Usage

### Premise:
* your model
* graphs (node features and edges)
* explanation
* your target sparsity (sparsity control)

### Format of Explanation

An explanation (a mask) is a list where each element
in is corresponding to an important mask for each class.

Then, the important mask is a `num_edges` size pytorch tensor.

### Evaluate

Given the inputs above, you can use the `ExplanationProcessor` and 
`XCollector` classes to obtain the results on the metrics.

#### Class ExplanationProcessor

It is for evaluating your model w/o the explanation. This class will generate
related probabilities for further metric calculations.

#### Class XCollector

This class is to collect the corresponding model output (the probabilities of
various label), then compute the Fidelity+(fidelity) and Fidelity-(fidelity_inv).

The example is given in `test`. Please refer to the `test/xgraph/test_metrics` function
where provides elaborated comments.