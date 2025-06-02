from computeranalysis import make_function, plot
import numpy as np

# create function with 2 variables
# R^2 -> R
f = make_function('sin(x+y)+ln(x^2-y^2)')
print(f)

# calculate derivative to x
df_x = f.df('x')
print(df_x)

# create parametrisation
# R -> R^2
r = make_function('t^2', '2-t+sqrt(t)')
print(r)

# combine
# R -> R
f_r = f.compose(r)
print(f_r)

# evaluate the function in a few points
for t in [9, 6, 2]:
    result = f_r.eval(t)
    print('value:', result)
    
g = make_function('4-x^2-y^2')
plot(g, xrange=np.linspace(-2,2), yrange=np.linspace(-2,2))

h = make_function('x^3')
plot(h, xrange=np.linspace(-2,2))