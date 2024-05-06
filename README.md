![Continuous Integration](https://github.com/markteffect/weightfactors/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
# **Weight Factors**
Calculate weight factors for survey data to approximate a representative sample


### **Installation**
```python
pip install weightfactors
```

or clone and install from source
```python
git clone https://github.com/markteffect/weightfactors
cd weightfactors
poetry install
```

### **Usage**
Currently, the package implements a generalized raking algorithm.  
If you'd like to see support for other algorithms, please open an issue or submit a pull request.  
  
Let's use the following dataset as an example:
```python
sample = pd.DataFrame(
    {
        "Gender": [
            "Male",
            "Male",
            "Female",
            "Female",
            "Female",
            "Male",
            "Female",
            "Female",
            "Male",
            "Female",
        ],
        "Score": [7.0, 6.0, 8.5, 7.5, 8.0, 5.0, 9.5, 8.0, 4.5, 8.5],
    }
)

```

Suppose our sample comprises 40% males and 60% females.  
If we were to calculate the average score, we would get:  
```python
np.average(sample["Score"])
# 7.25
```
Now, assuming a 50/50 gender distribution in the population,  
let's calculate weight factors to approximate the population distribution:  
```python
from weightfactors import GeneralizedRaker

raker = GeneralizedRaker({"Gender": {"Male": 0.5, "Female": 0.5}})
weights = raker.rake(sample)
# [1.25000008 1.25000008 0.83333334 0.83333334 0.83333334 1.25000008
# 0.83333334 0.83333334 1.25000008 0.83333334]
```

Let's calculate the average score again, this time applying the weight factors:  
```python
np.average(sample["Score"], weights=weights)
# 6.9791666284520835
```

For more detailed information and customization options, please refer to the docstrings.
