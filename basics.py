def chat(phrase):
    question=("why","how","when","where","who","what")
    capitalized = phrase.capitalize()
    if phrase.startswith(question):
        return "{}?".format(capitalized)
    else:
        return"{}.".format(capitalized)
print(chat("how are you"))

results = []
while True:
    user_input = input("say something:")
    if user_input == "end":
        break
    else:
        results.append(chat(user_input))
print(" ".join(results))