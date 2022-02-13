using CSV
using DataFrames
using MLDataUtils
using Random
using Statistics
using PyCall
using XGBoost
using EvalMetrics


#----------Load Data----------
data_dir = "../data/creditcard.csv"
df = DataFrame(CSV.File(data_dir))
println("Raw data:")
println(size(df))
println()

#----------Split Data for Training & Test----------
X = Array(df[!, Not(r"Class")])
X = transpose(X)
y = Array{Int64}(select(df, :Class))
y = vec(y)

Random.seed!(42);
(X_train, y_train), (X_test, y_test) = stratifiedobs((X, y), p = 0.7)
X_train = transpose(X_train)
X_test = transpose(X_test)
println("Size of training & test data:")
println(size(X_train))
println(size(X_test))
println(size(y_train))
println(size(y_test))
println()

#----------Scale Features----------
num_features = size(X_train)[2]
X_train_scaled = copy(X_train)
X_test_scaled = copy(X_test)
for col in 1:num_features
    feature_mean = mean(X_train[:, col])
    feature_std = std(X_train[:, col])
    X_train_scaled[:, col] = [i-feature_mean for i in X_train[:, col]] / feature_std
    X_test_scaled[:, col] = [j-feature_mean for j in X_test[:, col]] / feature_std
end

#----------Upsampling----------
ENV["PYTHON"] = "/usr/bin/python3"
imbo = pyimport("imblearn.over_sampling")
sm = imbo.SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train)
println("Size of data after SMOTE upsampling:")
println(size(X_train_sm))
println(size(y_train_sm))
println()

#----------Model Training----------
println("Training a XGBoost classifier ...")
dtrain = DMatrix(X_train_sm, label = y_train_sm)
y_test_vec = Vector(y_test)
dtest = DMatrix(X_test_scaled, label = y_test_vec)
param = [
    "eta" => 0.1,
    "objective" => "binary:logistic"
]


t = @time model = xgboost(dtrain, 1000, param=param, eval_set=dtest)
# 419.298300 seconds (1.13 M allocations: 61.997 MiB, 0.12% compilation time)
# 1000 iterations
println()

#----------Evaluation----------
y_pred_prob = XGBoost.predict(model, X_test_scaled)
println("Evaluate model performance ...")
println(binary_eval_report(y_test_vec, y_pred_prob))
println("Precision score @0.5:", precision(y_test_vec, y_pred_prob, 0.5))
println("Recall score @0.5: ", EvalMetrics.recall(y_test_vec, y_pred_prob, 0.5))

#Dict{String, Real}("precision@fpr0.05" => 0.03090909090909091, "recall@fpr0.05" => 0.918918918918919, "accuracy@fpr0.05" => 0.9499549407207144, "au_prcurve" => 0.7210605080264554, "samples" => 85443, "true negative rate@fpr0.05" => 0.9500087930124861, "au_roccurve" => 0.9715562681493392, "prevalence" => 0.0017321489179921118)
#Precision score @0.5:0.8531468531468531
#Recall score @0.5: 0.8243243243243243

