import random

unsure_answer = ["\nSorry, I'm not sure what you mean. Try entering one "
                 "of the following statements:\n 1) How are you today?\n 2) What does real love feel like?"]

known_response_dict = {
    'how': "I'm super duper thanks for askin trooper!",
    'what': "Love is like a milkshake after church on Sunday, delish."
}

print("Hello! My name is Sbeve and I'm your personal Assistant!")
while True:
    user_input = input()
    if 'how' in user_input.lower():
        print(known_response_dict['how'])
        break
    elif 'what' in user_input.lower():
        print(known_response_dict['what'])
        break
    else:
        print(unsure_answer[0])
