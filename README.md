# AI Handwritten Digit Recognizer (neural network + browser app)

**Application:** reads a handwritten digit 0-9 - the basis of cheque/amount reading, postal-code sorting,
exam-form (OMR) scanning and form digitisation.
**Data set:** scikit-learn `digits` (1797 images, 8x8 pixels, 10 classes), bundled - no download. For full MNIST,
change `load_data()` in `train_and_save.py` (and the 8x8 sizes in `digit_preprocess.py` / `app_template.html`).

## Run
    pip install -r requirements.txt
    python train_and_save.py     # model comparison + hyperparameter tuning + saves model and app.html
    python predict.py            # demo predictions (or: python predict.py my_digit.png)
    open app.html in any browser # draw a digit and see live predictions (works offline)

## Files
- train_and_save.py   compares 5 models, tunes the neural network (grid search, 5-fold CV), saves everything
- digit_preprocess.py crop + resize + centre step (identical to the browser app's)
- predict.py          command-line prediction
- app.html            self-contained drawing app (model weights embedded)
- app_template.html   app source (train_and_save.py fills in the weights)
- model/              model.joblib, weights.json, metrics.json (all metrics, confusion matrix, comparison)
