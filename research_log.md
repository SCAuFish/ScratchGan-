Mar. 23rd
Ran training overnight and it reached FID at about 0.03 (best one seen is 0.025). It generated some text that is readable although does not really make sense:
`A disappointed today is that the official would be allowing the West to make Germany lead and the move was thrown out until the wait .`
It has been trained on emnlp dataset which contains lots of sentences from news. The generated text therefore, can be hard to understand without a context.
If it is trained on context-independent text, such as tweets and short video comments, would the result be more qualitatively understandable?

Mar. 25th
Idea:
Can we use BERT or other general language model to guide the training of GAN?
Looking into Transformer structure and BERT.