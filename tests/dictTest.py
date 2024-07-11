import separation_detection

a = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

b = {
    "name": "Mike",
    "age": 35,
    "city": "New York"
}


k = {}

k[0] = a
k[1] = b

print(list(a.keys()))