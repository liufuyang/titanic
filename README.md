
# Predict survival on the Titanic using Python

Just another amateur solution to Kaggle's titanic competition

### Task Description:

Based on some personal info, predict survived or not.

### Variable Description:

Variable| Description
--- | ---
|survival       |Survival (0 = No; 1 = Yes)
|pclass          |Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
|name            |Name
|sex             |Sex
|age             |Age
|sibsp           |Number of Siblings/Spouses Aboard
|parch           |Number of Parents/Children Aboard
|ticket          |Ticket Number
|fare            |Passenger Fare
|cabin           |Cabin
|embarked        |Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

#### SPECIAL NOTES:
* `Pclass` is a proxy for socio-economic status (SES)
 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

* `Age` is in Years; Fractional if Age less than One (1)
 If the Age is Estimated, it is in the form xx.5

With respect to the family relation variables (i.e. `sibsp` and `parch`)
some relations were ignored.  The following are the definitions used
for sibsp and parch.

* `Sibling`:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
* `Spouse`:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
* `Parent`:   Mother or Father of Passenger Aboard Titanic
* `Child`:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

Other family relatives excluded from this study include cousins,
nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
only with a nanny, therefore parch=0 for them.  As well, some
travelled with very close friends or neighbors in a village, however,
the definitions do not support such relations.

# Current Score:

## Current Kaggle public score: 0.8134

Reached by
```
feature = ['Pclass','FamilySurvived', 'FamilyDied',
          'Title_s_Master', 'Title_s_Miss', 'Title_s_Mr', 'Title_s_Mrs',
          ]
clf = RandomForestClassifier(n_estimators=1000,
                                       criterion='entropy',
                                       random_state=1,
                                       min_samples_split=2,
                                       min_samples_leaf=2,
                                       max_features='auto',
                                       bootstrap=True,
                                       oob_score=True,
                                       max_depth=3,
                                       n_jobs=-1)
```
With `max_depth=4` the Kaggle score gets lower, however the local CV score gets a little higher.

Check Demo2 ipython notebook for details.
