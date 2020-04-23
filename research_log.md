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

## Mar. 29th
1. Tried using BERT for easy task: sentiment analysis. But it "cleverly" learned to give 0.5 on all the inputs since the dataset is balanced so that the loss could be minimized. Interesting.

## Mar. 30th
1. The added dense linear layer is not in "trainable_weights"? Investigate about that. Also add some more layers on top and maybe remove the sigmoid at the end.
Followup: the trainable weights are there. It is just there need to be input through it before its initialization

## Apr. 1st
1. Huge discovery when re-reading the BERT paper: in the paper's setting, when predicting the masked word, the vector input to softmax classification layer is the vector corresponding to the position of the masked word. On the other hand, the pooled single-vector output was used to accomplish "Next Sentence Prediction" task. Let's try with the other input.

2. Noticed that in the BERT paper, they reached pretty high accuracy (~90%) on `Cloze` test. Trying to reimplement the word predictor to see if it works better.

TODO:
1. In BERT_cloze, the model is already implemented, try running sanity test and start training

Q:
1, GAN - a special version of Actor-Critic?
2, BERT: a critic or environment

## Apr. 4th
1. Rewrite the Word Predictor in BERT_cloze, implemented some training optimization. After one epoch of training, it is giving some reasonable outputs (although not the same word, POS is basically correct). Trying to run more epochs to see the result.

2. TODO: use BERT as discriminator and only output whether the sentence is generated or true sentences.

## Apr. 5th
1. Train BERT_Cloze for 10 epochs and it reaches pretty decent word prediction result. For example when asked to predict which word fits in `"the [] has caused panic around the world"`, top five candidates are ['incident']['attack']['case']['shooting']['virus'].

2. As a result of above, maybe we can try with BERT text generator by giving all mask words at first?

## Apr. 6th 
1. Now training BERT_Generator to complete a sentence given the previous part of the sentence.

## Apr. 11th
1. Using BERT to directly generate a sentence does not perform very well. It starts with a sentence covered all by masks, and generate word one by one revealing the word under the mask. A possible modification: first generate the word on the position with highest confidence? So that the word generation does not have to follow the order from left to right.

2. Let's first make the sentence scorer, scoring each word choice in the sentence.

3. During training, as shown in line 202 of experiments.py, output from discriminator per word is inputted to calculate REINFORCE loss. Is it possible to pass in BERT scorer output at the same place?
Sounds doable, needs to look into reinforce_loss function in losses.py.

4. Since the codes were for python2, many functions need to be refactored. Like xrange() should be changed to range()

## Apr. 13th
1. Trying to integrate the critic with their ScratchGan. Since their codes are all in python2 and tensorflow 1.x, decided to move to those versions. Need some refactor to make it work.

## Apr. 14th
1. In BERT_Cloze_Py2, I have successfully run a BERT word predictor in tensorflow 1.14 and python 2 fashion using Graph execution. (So smart)

## Apr. 21st
1. Tring to integrate bert discriminator into ScratchGan. Here are some notes:
    disc_logits returned from discriminator: `Tensor("LSTMEmbedDiscNet_1/mul_2:0", shape=(500, 52), dtype=float32)`

    500 is batch size. 52 should be max sequence length

    gen_output from generator: `{'sequence_length': <tf.Tensor 'lstm_gen/Minimum:0' shape=(500,) dtype=int32>, 'logprobs': <tf.Tensor 'lstm_gen/mul_1:0' shape=(500, 52) dtype=float32>, 'sequence': <tf.Tensor 'lstm_gen/mul:0' shape=(500, 52) dtype=int32>}`

2. It is too complex to integrate it in graph execution mode. The key problem now is, we need to transform indices of scratchgan to indices of bert module. Due to graph execution, we cannot use regular way with dict. In jupyter notebook we showed that there is some way to first eval, and then deal with the result without hurting the graph. Should try this way.

3. idk. Maybe the current way works? I notice that loading from checkpoint is causing a lot of issue. Maybe because I always ctrl+C to end the program. If nothing else, delete checkpoint files and run.

## Apr. 22nd
1. Okay now it runs. But for now I am simply adding word-wise score from bert to disc_fake logits. Need to take a look into whether that makes sense.
