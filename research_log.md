## Mar. 23rd
Ran training overnight and it reached FID at about 0.03 (best one seen is 0.025). It generated some text that is readable although does not really make sense:
`A disappointed today is that the official would be allowing the West to make Germany lead and the move was thrown out until the wait .`

It has been trained on emnlp dataset which contains lots of sentences from news. The generated text therefore, can be hard to understand without a context.

If it is trained on context-independent text, such as tweets and short video comments, would the result be more qualitatively understandable?


## Mar. 25th
Idea:

Can we use BERT or other general language model to guide the training of GAN?

Looking into Transformer structure and BERT.

## Mar. 26th
1, In bert_research folder, managed to run bert on SQUAD problem. After several minutes' finetuning, it can already reach accuracy of about 83%

2, In BERT folder, using bert from tensorflow hub to making output. It works.

3, Now implementing functions and additional layers to finetune BERT to see how well it can be on predicting missing words.

TODO:
see #3 above

## Mar. 27th
1, BERT + masked word predictor was implemented. Now training. Need to check how it works.

2, If the accuracy is not high in the predictor, print out the words and check the best several candidates to see if most candidates make sense.

3, To add this critic into GAN, basically check train() function in scratchgan/experiments.py, in which result from discriminator is backpropagated into generator.

TODO:

1, Qualitatively check how BERT predicts the missing word

2, Add critic to GAN training

## Mar. 28th
1, At first, BERT is predicting frequent words to achieve lower loss (e.g. "the", "a", "," due to unbalanced data)

2, After making use of weighted cross entropy, it is giving some random words ... 

3, Trying to shrink the size of training data to see that it at least overfits; 

4, In segment encoding passed into BERT, the masked word is labeled 1. Hopefullly it now knows that it should fill in the blank for that word

TODO:
5, Implement some easy functions with BERT, such as sentiment analysis