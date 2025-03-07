# Many rules are missing
# Multiplication of two Paulis
mul = {}
mul["I", "I"] = [(1, "I")]
mul["I", "X"] = [(1, "X")]
mul["I", "Y"] = [(1, "Y")]
mul["I", "Z"] = [(1, "Z")]
mul["X", "I"] = [(1, "X")]
mul["X", "X"] = [(1, "I")]
mul["X", "Y"] = [(1j, "Z")]
mul["X", "Z"] = [(-1j, "Y")]
mul["Y", "I"] = [(1, "Y")]
mul["Y", "X"] = [(-1j, "Z")]
mul["Y", "Y"] = [(1, "I")]
mul["Y", "Z"] = [(1j, "X")]
mul["Z", "I"] = [(1, "Z")]
mul["Z", "X"] = [(1j, "Y")]
mul["Z", "Y"] = [(-1j, "X")]
mul["Z", "Z"] = [(1, "I")]

com = {}
# Single Paulis
com["X", "I"] = []
com["X", "X"] = []
com["X", "Y"] = [(2j, "Z")]
com["X", "Z"] = [(-2j, "Y")]

# Double Paulis
com["ZZ", "II"] = []
com["ZZ", "IX"] = [(2j, "ZY")]
com["ZZ", "IY"] = [(-2j, "ZX")]
com["ZZ", "IZ"] = []
com["ZZ", "XI"] = [(2j, "YZ")]
com["ZZ", "XX"] = []
com["ZZ", "XY"] = []
com["ZZ", "XZ"] = [(2j, "YI")]
com["ZZ", "YI"] = [(-2j, "XZ")]
com["ZZ", "YX"] = []
com["ZZ", "YY"] = []
com["ZZ", "YZ"] = [(-2j, "XI")]
com["ZZ", "ZI"] = []
com["ZZ", "ZX"] = [(2j, "IY")]
com["ZZ", "ZY"] = [(-2j, "IX")]
com["ZZ", "ZZ"] = []

# Dissipation part of Lindbladian
dis = {}
dis["M", "I"] = []
dis["M", "X"] = [(-1/2, "X")]
dis["M", "Y"] = [(-1/2, "Y")]
dis["M", "Z"] = [(-1, "I"), (-1, "Z")]

# Merge
rules = {}
rules = {"mul": mul, "com": com, "dis": dis}
