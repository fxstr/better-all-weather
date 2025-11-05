from sklearn.linear_model import LinearRegression

regression = LinearRegression()

def get_relative_slope(data):
    '''
    Calculates the relative slope of a linear regression line for the given data. Returns a
    percent value (annualized expected return)
    data is a df where rows are dates and columns are instruments.
    '''

    def calculate_slope(series):
        # Gives a list of days that have passed since the first day, for every row 
        time_delta = (series.index - series.index[0]).to_series().dt.days
        y = series.values.reshape(-1, 1)
        x = time_delta.to_numpy().reshape(-1, 1)
        regression.fit(x, y)
        intercept = regression.intercept_[0]
        slope = regression.coef_[0][0]
        return ((intercept + slope * 365) / intercept) - 1
    
    return data.apply(calculate_slope)