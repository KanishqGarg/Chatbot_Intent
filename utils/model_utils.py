from sklearn.linear_model import LogisticRegression
def build_model(x,y):
    model = LogisticRegression(random_state=0, max_iter=10000)
    model.fit(x,y)
    return model