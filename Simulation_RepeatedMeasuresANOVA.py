import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from statistics import mean

def simulate_repeated_anova(means, cov_matrix, repetition, repetition_cv,
                            folds_cv, sample_size, seed, detailed = False):

  #This initiates a dictionary that will be the result table in the end
  result_dict = {"Repetition":[], "RM-ANOVA":[], "Greenhouse-Geisser":[],
                 "Huynh-Feldt":[], "Cross-Validation":[]}

  #To make the results replicable, a seed for the pseudorandomly
  #generated data is set
  rnd_seed = seed


  #For every repetition, this loop over data generation & model computation
  for n in range(repetition):

    np.random.seed(rnd_seed)
    result_dict["Repetition"].append(n+1)

    #(Pseudo-)Random data generation
    #First, it is checked which type of distribution was specified
    #then, the function will loop over the list of touples and create random
    #data for each repeated measure, which is then stored in a dictionary called
    #"generate_dict")

    #Data generation from multivariate normal distribution
    generate_dict = {}
    generate = np.random.multivariate_normal(means, cov_matrix, sample_size)

    #Store data for each measurement in a dictionary for RM ANOVA
    count = 1
    for measurement in generate.T:
      generate_dict[f"Measurement{count}"] = measurement
      count += 1


    #Repeated Measures ANOVA (Standard Analysis)
    #Number of repeated measures
    n_measures = len(generate_dict.keys())

    #Calculate grand mean
    grand_mean = np.array([generate_dict[k] for k in generate_dict]).mean()

    #Calculate sums of squares
    #SS Total
    ss_total = 0
    for n in generate_dict.keys():
      for m in generate_dict[n]:
        ss_total += (m-grand_mean)**2

    #SS Subject
    sum_squared = 0
    for n in range(sample_size):
      subject_sum = 0
      for key in generate_dict:
        subject_sum += generate_dict[key][n]
      sum_squared += ((subject_sum/n_measures)-grand_mean)**2
    ss_subject = n_measures*sum_squared

    #SS Measure
    sum_measure = 0
    for measure in generate_dict.keys():
      sum_measure += (np.mean(generate_dict[measure])-grand_mean)**2
    ss_measure = sum_measure*sample_size

    #SS Error
    ss_error = ss_total - ss_subject - ss_measure

    #Degrees of Freedom
    df_subject = sample_size-1
    df_measure = n_measures-1
    df_error = df_subject*df_measure
    df_total = sample_size*n_measures - 1

    #MS
    ms_measure = ss_measure/df_measure
    ms_error = ss_error/df_error

    #F-test & p-value
    F = ms_measure/ms_error
    p_value = stats.f.sf(F,df_measure,df_error,loc=0,scale=1)

    #append p-value to the result dictionary
    result_dict["RM-ANOVA"].append(p_value)


    #Correction Methods

    #Greenhouse-Geisser (GG) Correction
    #Create Covariance Matrix
    measures_list = [generate_dict[n] for n in generate_dict.keys()]
    measures_array = np.array(measures_list)
    m_cov = np.cov(measures_array)

    #Total Mean for all measures
    total_mean = sum([row.mean() for row in m_cov])/len(m_cov)

    #Double Centering
    t=0
    for n in range(len(m_cov)):
      for l in range(len(m_cov)):
        t += ((m_cov[n][l]-total_mean)-(m_cov[n].mean()-total_mean)-(m_cov[l].mean()-total_mean))**2
    s=0
    for n in range(len(m_cov)):
      s += m_cov[n][n]-total_mean-2*(m_cov[n].mean()-total_mean)
      u = s**2

    #Epsilon (degree of sphericity)
    epsilon = u/((len(m_cov)-1)*t)

    #Adjusted degrees of freedom
    v1 = (len(m_cov)-1)*epsilon
    v2 = v1*(sample_size-1)

    #P-Value for GG
    p_value_gg = stats.f.sf(F,v1, v2,loc=0,scale=1)

    #append p_value_gg to result dictionary
    result_dict["Greenhouse-Geisser"].append(p_value_gg)


    #Huynh-Feldt Correction
    #Adjusted Epsilon and adjusted degrees of freedom
    epsilon_hf = (sample_size*(len(m_cov)-1)*epsilon-2)/((len(m_cov)-1)*(sample_size-1-((len(m_cov)-1)*epsilon)))
    v1_hf = (len(m_cov)-1)*epsilon_hf
    v2_hf = v1_hf*(sample_size-1)

    p_value_hf = stats.f.sf(F,v1_hf, v2_hf,loc=0,scale=1)

    #Append p_value_hf to result dictionary
    result_dict["Huynh-Feldt"].append(p_value_hf)


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

    #For cross-validation of the Sklearn library a dataframe is created
    rm_dict = {}
    rm_dict["Subject"] = []
    rm_dict["Measurement"] = []
    rm_dict["Score"] = []

    for n in range(len(generate)):
      count = 1
      for m in generate[n]:
        rm_dict["Subject"].append(n+1)
        rm_dict["Measurement"].append(count)
        rm_dict["Score"].append(m)
        count += 1

    rm_df = pd.DataFrame(rm_dict)

    #X needs to be passed as a dummy variable to regression
    X = pd.get_dummies(rm_df["Measurement"])

    #Initiate a win count (counts win for every cv-repetition)
    wins = 0

    #For every repetition run cross-validation with mean & factor model
    #Note: GroupShuffleSplit is just a splitter. By setting the test_size equal
    #to 0.1 for 10 folds, and 0.2 for 5 folds, we get a 10-fold and 5-fold cv,
    #respectively.
    if folds_cv == 10:
      test_size = 0.1
    else:
      test_size = 0.2

    #For every repetition run cross-validation with mean & factor model
    for n in range(repetition_cv):

      #We use grouped cross-validation --> GroupShuffleSplit
      gss = GroupShuffleSplit(n_splits = folds_cv, test_size = test_size,
                              random_state = n)

      score1 = np.sqrt(cross_val_score(MeanPredictor(), X, rm_df["Score"],
                                       groups = rm_df["Subject"],
                                       scoring='neg_mean_squared_error',
                                       cv=gss, n_jobs=-1).mean()*-1)

      score2 = np.sqrt(cross_val_score(LinearRegression(), X, rm_df["Score"],
                                       groups = rm_df["Subject"],
                                       scoring='neg_mean_squared_error',
                                       cv=gss, n_jobs=-1).mean()*-1)

      #Compare scores
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
                            columns=["RM-ANOVA", "Greenhouse-Geisser",
                                     "Huynh-Feldt", "Cross-Validation"],
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

    #Sort table by the p-values for RM ANOVA in descending order
    result_df = result_df.sort_values("RM-ANOVA", ascending=False)

    #Apply all styles to the table
    result_df = (result_df.style
      .hide_index()
      .applymap(color_significant, subset=["RM-ANOVA", "Greenhouse-Geisser",
                                           "Huynh-Feldt"])
      .applymap(factor_color, subset=["Cross-Validation"])
      .format({"RM-ANOVA": "{:.4f}", "Greenhouse-Geisser": "{:.4f}",
               "Huynh-Feldt": "{:.4f}"}))

    return result_df

  #If detailed = False, the relative frequencies of significant results (for
  #ANOVA and Welch) and factor-model wins (CV) are returned
  else:
    ratio_dict = {}
    ratio_dict["RM-ANOVA"] = sum(i <= 0.05 for i in result_dict["RM-ANOVA"])/repetition
    ratio_dict["Greenhouse-Geisser"] = sum(i <= 0.05 for i in result_dict["Greenhouse-Geisser"])/repetition
    ratio_dict["Huynh-Feldt"] = sum(i <= 0.05 for i in result_dict["Huynh-Feldt"])/repetition
    ratio_dict["Cross-Validation"] = sum(i == "Factor" for i in result_dict["Cross-Validation"])/repetition

    return ratio_dict


#Define parameters
means = [[1, 1, 1], [1, 1.1, 1.2]]

cov1 = np.array([[0.25, 0.1, 0.1],
                [0.1, 0.25, 0.1],
                [0.1, 0.1, 0.25]])

cov2 = np.array([[1, 0.4, 0.4],
                [0.4, 1, 0.4],
                [0.4, 0.4, 1]])

cov3 = np.array([[0.25, 0.1, 0.2],
                [0.1, 0.25, 0.2],
                [0.2, 0.2, 1]])

cov4 = np.array([[0.25, 0.1, 0.025],
                [0.1, 0.25, 0.1],
                [0.025, 0.1, 0.25]])

cov5 = np.array([[1, 0.4, 0.1],
                [0.4, 1, 0.4],
                [0.1, 0.4, 1]])

cov6 = np.array([[0.25, 0.1, 0.05],
                [0.1, 0.25, 0.2],
                [0.05, 0.2, 1]])

cov_list = [cov1, cov2, cov3, cov4, cov5, cov6]

sample_sizes = [20, 50, 100]

#Initiate dictionary for the results
simulation_dict = {}
simulation_dict["Means"] = []
simulation_dict["Sample Size"] = []
simulation_dict["Cov_Matrix"] = []
simulation_dict["RM ANOVA"] = []
simulation_dict["GG"] = []
simulation_dict["HF"] = []
simulation_dict["Cross-Validation"] = []

#Run simulations by looping over parameters
for mean in means:
  for cov in cov_list:
    for size in sample_sizes:

      if size == 100:
        #This runs the simulation and stores it in a variable
        simulation = simulate_repeated_anova(means=mean, cov_matrix=cov, folds_cv=10,
                                repetition=100, repetition_cv=200,
                                sample_size=size, seed=1, detailed = False)

        #Pass results to simulation dictionary
        simulation_dict["Means"].append(mean)
        simulation_dict["Cov_Matrix"].append(cov)
        simulation_dict["Sample Size"].append(size)
        simulation_dict["RM ANOVA"].append(simulation["RM-ANOVA"])
        simulation_dict["GG"].append(simulation["Greenhouse-Geisser"])
        simulation_dict["HF"].append(simulation["Huynh-Feldt"])
        simulation_dict["Cross-Validation"].append(simulation["Cross-Validation"])

      else:
        #This runs the simulation and stores it in a variable
        simulation = simulate_repeated_anova(means=mean, cov_matrix=cov, folds_cv=5,
                                repetition=100, repetition_cv=200,
                                sample_size=size, seed=1, detailed = False)

        #Pass results to simulation dictionary
        simulation_dict["Means"].append(mean)
        simulation_dict["Cov_Matrix"].append(cov)
        simulation_dict["Sample Size"].append(size)
        simulation_dict["RM ANOVA"].append(simulation["RM-ANOVA"])
        simulation_dict["GG"].append(simulation["Greenhouse-Geisser"])
        simulation_dict["HF"].append(simulation["Huynh-Feldt"])
        simulation_dict["Cross-Validation"].append(simulation["Cross-Validation"])

#Create dataframe to display as final table
simulation_df = pd.DataFrame(
    data = simulation_dict,
    columns = ["Means", "Cov_Matrix", "Sample Size", "RM ANOVA", "GG", "HF",
               "Cross-Validation"])

#To have the full table printed (and results with 2 decimals)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.options.display.float_format = "{:,.2f}".format

print(simulation_df)
