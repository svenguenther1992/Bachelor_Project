import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from statistics import mean

#One-Way ANOVA Simulation Code

#The function expects a list of tuples as input, for which each touple refers
#to a category of the independent variable. The first entry in each tuple
#corresponds to the mean and the second entry corresponds to the
#standard deviation of the category.

def simulate_anova(data, distribution, repetition, repetition_cv, folds_cv,
                   sample_size, seed, detailed = False):

  #This initiates a dictionary that will be the result table in the end
  result_dict = {"Repetition":[], "ANOVA":[], "Welch":[],
                   "Cross-Validation":[]}

  #to make the results replicable, a seed for the pseudorandomly
  #generated data is set
  rnd_seed = seed


  #For every repetition, this loops over the data generation & model computation
  for n in range(repetition):

    np.random.seed(rnd_seed)
    result_dict["Repetition"].append(n+1)

    #(Pseudo-)Random data generation
    #First, it is checked which type of distribution was specified
    #then, the function will loop over the list of touples and create random
    #data for each category, which is then stored in a dictionary called
    #"generate_dict")

    #data generation from normal distribution
    if distribution == "normal":
      generate_dict = {}
      count = 1

      for touple in data:
        generate_dict[f"category {count}"] = np.random.normal(loc=touple[0],
                                                            scale=touple[1],
                                                            size=sample_size)
        count += 1

    #data generation from skewed distribution (lognormal)
    elif distribution == "skewed":
      generate_dict = {}
      count = 1

      for touple in data:
        generate_dict[f"category {count}"] = np.random.lognormal(mean=touple[0],
                                                            sigma=touple[1],
                                                            size=sample_size)
        count += 1


    #One-Way ANOVA (Standard Analysis)
    #Calculate Grand Mean
    grand_mean = np.array([generate_dict[k] for k in generate_dict]).mean()

    #Sum of Squares
    #SS Total
    ss_total = 0
    for n in generate_dict.keys():
      for m in generate_dict[n]:
        ss_total += (m-grand_mean)**2

    #SS Between
    ss_between = 0
    for n in generate_dict.keys():
      ss_between += len(generate_dict[n])*(generate_dict[n].mean() - grand_mean)**2

    #SS Within (Error)
    ss_within = ss_total - ss_between

    #Degrees of Freedom
    df_between = len(generate_dict.keys())-1
    df_within = sum(len(generate_dict[n]) for n in generate_dict.keys())- len(generate_dict.keys())

    #Mean Squares
    ms_between = ss_between/df_between
    ms_within = ss_within/df_within

    #F-Test
    F = ms_between/ms_within

    #p-value
    p_value = stats.f.sf(F,df_between,df_within,loc=0,scale=1)

    #append p-value to the result dictionary
    result_dict["ANOVA"].append(p_value)


    #WELCH ANOVA
    #Calculate group weights (n/s²)
    weights = {}
    for group in generate_dict.keys():
      n = len(generate_dict[group])
      s_squared = sum((x-generate_dict[group].mean())**2 for x in generate_dict[group])/(n-1)
      weights[group] = n/s_squared

    #Calculate welch grand mean (sum of mean of each group * group weight,
    #divided by total sum of weights)
    sum_mean_weight = 0
    sum_weights = 0
    for group in generate_dict.keys():
      sum_mean_weight += generate_dict[group].mean() * weights[group]
      sum_weights += weights[group]
    welch_mean = sum_mean_weight/sum_weights

    #Calculate sums of squares (sum of group weight * (group mean - welch mean)²)
    ss_welch = 0
    for group in generate_dict.keys():
      ss_welch += weights[group]*((generate_dict[group].mean()-welch_mean)**2)

    #Calculate mean squares (ss_welch/k-1)
    ms_welch = ss_welch/(len(generate_dict.keys())-1)

    #Calculate lambda
    z = 0
    for group in generate_dict.keys():
      y = ((1-(weights[group]/sum_weights))**2)/(len(generate_dict[group])-1)
      z += y
    lamb = 3*z/(len(generate_dict.keys())**2-1)

    #Calculate F
    F_welch = ms_welch/(1+(2*lamb*(len(generate_dict.keys())-2)/3))

    #p-value
    p_welch = stats.f.sf(F_welch,len(generate_dict.keys())-1,1/lamb,loc=0,scale=1)

    #append p-value to result dictionary
    result_dict["Welch"].append(p_welch)


    #Cross-Validation approach
    #Prediction by Mean (to pass to sklearn cross-validation function)
    #Note: this is not necessary for regression as this model exists already
    #in the Sklearn package
    class MeanPredictor():

      def __init__(self, mean=None):
        if mean is not None:
          self.mean =mean
        else:
          self.mean=None

      def fit(self, X, Y):
        self.mean = np.mean(Y)

      def predict(self, X):
        return np.full(X.shape[0], self.mean)

      def get_params(self, deep=False):
        return {"mean":self.mean}


    #Models for comparison (LinearRegression is from Sklearn library)
    models = [("Prediction by Mean", MeanPredictor()),
              ("Linear Regression", LinearRegression())]

    #For the cross-validation of the sklearn library a dataframe is created
    #because the data is expected to be in a certain form when passed
    #Create dataframe
    x_list = []
    y_list = []
    for key in generate_dict.keys():
      for n in generate_dict[key]:
        x_list.append(key)
        y_list.append(n)
    df = pd.DataFrame(list(zip(x_list, y_list)), columns=["x","y"])

    #x needs to be passed as a dummy variable
    X = pd.get_dummies(df["x"])

    #initiate a win count (counts win for every cv-repetition)
    wins = 0

    for n in range(repetition_cv):
      cv = KFold(n_splits=folds_cv, shuffle=True, random_state=n)
      score1 = np.sqrt(cross_val_score(models[0][1], X, df["y"], scoring='neg_mean_squared_error', cv=cv, n_jobs=-1).mean()*-1)
      score2 = np.sqrt(cross_val_score(models[1][1], X, df["y"], scoring='neg_mean_squared_error', cv=cv, n_jobs=-1).mean()*-1)

      #compare scores
      if score1 < score2:
        wins += -1
      elif score1 > score2:
        wins += 1
      else:
        wins = wins

    #Depending on which model wins, append the winner to the result table
    if wins > 0:
      result_dict["Cross-Validation"].append("Factor")
    elif wins < 0:
      result_dict["Cross-Validation"].append("Mean")
    else:
      result_dict["Cross-Validation"].append("Even")


    #To not generate the same data twice, the seed is changed for the next rep.
    rnd_seed += 1

  #AFTER all repetitions are finished:
  #If detailed = True, a table is constructed that displays the results for
  #every repetition
  if detailed == True:

    #This puts the result table together that is displayed in the end
    result_df = pd.DataFrame(result_dict,
                            columns=["ANOVA", "Welch", "Cross-Validation"],
                            index = result_dict["Repetition"])

    #Table formatting
    #The following two functions color significant/factor model results
    def color_significant(value):
      if value > 0.05:
        color = "red"
      else:
        color = "green"
      return "color: %s" % color

    def factor_color(model):
      if model == "Mean":
        color = "red"
      else:
        color = "green"
      return "color: %s" % color

    #The table is sorted by the p-values in descending order
    result_df = result_df.sort_values("ANOVA", ascending=False)

    #Applying all styles to the table
    result_df = (result_df.style
      .hide_index()
      .applymap(color_significant, subset=["ANOVA", "Welch"])
      .applymap(factor_color, subset=["Cross-Validation"])
      .format({"ANOVA": "{:.4f}", "Welch": "{:.4f}"}))

    return result_df

  #If detailed = False, the relative frequencies of significant results (for
  #ANOVA and Welch) and factor-model wins (CV) are returned
  else:
    ratio_dict = {}
    ratio_dict["ANOVA"] = sum(i <= 0.05 for i in result_dict["ANOVA"])/repetition
    ratio_dict["Welch"] = sum(i <= 0.05 for i in result_dict["Welch"])/repetition
    ratio_dict["Cross-Validation"] = sum(i == "Factor" for i in result_dict["Cross-Validation"])/repetition

    return ratio_dict


#Iterative Simulation Run with unequal means --> power
#Define all parameters
distribution = ["normal", "skewed"]
std_dev = [[0.5, 0.5, 0.5], [1, 1, 1], [0.5, 0.5, 1]]
sample_sizes = [20, 50, 100]

#Initiate dictionary for the results
simulation_dict_unequal = {}
simulation_dict_unequal["Distribution"] = []
simulation_dict_unequal["Std. Dev."] = []
simulation_dict_unequal["Sample Size"] = []
simulation_dict_unequal["ANOVA"] = []
simulation_dict_unequal["Welch"] = []
simulation_dict_unequal["Cross-Validation"] = []

#Loop over the parameters and run the simulation
for a in distribution:
  for b in range(len(std_dev)):
    for c in sample_sizes:
      data = [(1, std_dev[b][0]), (1.2, std_dev[b][1]), (1.4, std_dev[b][2])]

      if c != 100:
        #This runs the simulation and stores it in a variable
        simulation = simulate_anova(data, distribution = a, repetition = 100,
                repetition_cv = 200, folds_cv = 5, sample_size = c, seed = 1,
                detailed = False)

        #Pass results to simulation dictionary
        simulation_dict_unequal["ANOVA"].append(simulation["ANOVA"])

        simulation_dict_unequal["Welch"].append(simulation["Welch"])

        simulation_dict_unequal["Cross-Validation"].append(simulation["Cross-Validation"])

      else:
        #This runs the simulation and stores it in a variable
        simulation = simulate_anova(data, distribution = a, repetition = 100,
                repetition_cv = 200, folds_cv = 10, sample_size = c, seed = 1,
                detailed = False)

        #Pass results to simulation dictionary
        simulation_dict_unequal["ANOVA"].append(simulation["ANOVA"])

        simulation_dict_unequal["Welch"].append(simulation["Welch"])

        simulation_dict_unequal["Cross-Validation"].append(simulation["Cross-Validation"])


      simulation_dict_unequal["Distribution"].append(a)

      simulation_dict_unequal["Std. Dev."].append(std_dev[b])

      simulation_dict_unequal["Sample Size"].append(c)

#Create a dataframe to display the results
simulation_df_unequal = pd.DataFrame(
    data = simulation_dict_unequal,
    columns = ["Distribution", "Std. Dev.", "Sample Size", "ANOVA", "Welch",
               "Cross-Validation"])

pd.options.display.float_format = "{:,.2f}".format

print(simulation_df_unequal)


#Iterative Simulation Run with equal means --> Type 1 Error rate
distribution = ["normal", "skewed"]
std_dev = [[0.5, 0.5, 0.5], [1, 1, 1], [0.5, 0.5, 1]]
sample_sizes = [20, 50, 100]

simulation_dict_equal = {}
simulation_dict_equal["Distribution"] = []
simulation_dict_equal["Std. Dev."] = []
simulation_dict_equal["Sample Size"] = []
simulation_dict_equal["ANOVA"] = []
simulation_dict_equal["Welch"] = []
simulation_dict_equal["Cross-Validation"] = []

for a in distribution:
  for b in range(len(std_dev)):
    for c in sample_sizes:
      data = [(1, std_dev[b][0]), (1, std_dev[b][1]), (1, std_dev[b][2])]

      if c != 100:
        #This runs the simulation and stores it in a variable
        simulation = simulate_anova(data, distribution = a, repetition = 100,
                repetition_cv = 200, folds_cv = 5, sample_size = c, seed = 1,
                detailed = False)

        #Pass results to simulation dictionary
        simulation_dict_equal["ANOVA"].append(simulation["ANOVA"])

        simulation_dict_equal["Welch"].append(simulation["Welch"])

        simulation_dict_equal["Cross-Validation"].append(simulation["Cross-Validation"])

      else:
        #This runs the simulation and stores it in a variable
        simulation = simulate_anova(data, distribution = a, repetition = 100,
                repetition_cv = 200, folds_cv = 10, sample_size = c, seed = 1,
                detailed = False)

        #Pass results to simulation dictionary
        simulation_dict_equal["ANOVA"].append(simulation["ANOVA"])

        simulation_dict_equal["Welch"].append(simulation["Welch"])

        simulation_dict_equal["Cross-Validation"].append(simulation["Cross-Validation"])

      simulation_dict_equal["Distribution"].append(a)

      simulation_dict_equal["Std. Dev."].append(std_dev[b])

      simulation_dict_equal["Sample Size"].append(c)

simulation_df_equal = pd.DataFrame(
    data = simulation_dict_equal,
    columns = ["Distribution", "Std. Dev.", "Sample Size", "ANOVA", "Welch",
               "Cross-Validation"])

print(simulation_df_equal)
