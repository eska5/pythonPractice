from unittest import result


def solution(A):
    listOfCorrectParts = []

    #checking if the words are correct on its own and appending them to 
    for wordIndex in range(len(A)):
        checkIfUniqueLetters = True
        for letterIndexFirst in range(len(A[wordIndex])):
            for letterIndexSecond in range(letterIndexFirst+1,len(A[wordIndex])):
                if(A[wordIndex][letterIndexFirst] == A[wordIndex][letterIndexSecond]):
                    checkIfUniqueLetters = False
                    break
            if checkIfUniqueLetters == False:
                break
        if checkIfUniqueLetters == True:
            listOfCorrectParts.append(A[wordIndex])

    #correct words are in a list at this point
    #with sorted letters

    for index in range(len(listOfCorrectParts)):
        listOfCorrectParts[index] = ''.join(sorted(listOfCorrectParts[index]))

    #sorted

    result = 0
    for word in listOfCorrectParts:
        if result < len(listOfCorrectParts):
            result = len(listOfCorrectParts)
    
    listOfCorrectParts2 = listOfCorrectParts
    changes = -1
    while changes !=0 :
        changes = 0
        listOfCorrectParts3 = []
        for wordIndexOne in range(len(listOfCorrectParts)):
            for wordIndexTwo in range(len(listOfCorrectParts2)):
                wordIndex1 = len(listOfCorrectParts[wordIndexOne])
                wordIndex2 = len(listOfCorrectParts2[wordIndexTwo])
                iter1 = 0
                iter2 = 0
                newWord = ""
                tempResult = 0
                errorRes= False
                while iter1 + iter2 < wordIndex1 + wordIndex2 - 2 :
                    if listOfCorrectParts[wordIndexOne][iter1] ==  listOfCorrectParts2[wordIndexTwo][iter2]:
                        errorRes = True
                        break
                    elif listOfCorrectParts[wordIndexOne][iter1] <  listOfCorrectParts2[wordIndexTwo][iter2]:
                        newWord = newWord + (listOfCorrectParts[wordIndexOne][iter1])
                        iter1+=1
                        tempResult+=1
                    elif listOfCorrectParts[wordIndexOne][iter1] >  listOfCorrectParts2[wordIndexTwo][iter2]:
                        newWord = newWord + (listOfCorrectParts2[wordIndexTwo][iter2])
                        iter2+=1
                        tempResult+=1

                    if iter1 == wordIndex1:
                        for iter in range(iter2,wordIndex2):
                            newWord+= (listOfCorrectParts2[wordIndexTwo][iter])
                            tempResult+=1
                        iter2 = wordIndex2
                    elif iter2 == wordIndex2:
                        for iter in range(iter1,wordIndex1):
                            newWord+= (listOfCorrectParts[wordIndexOne][iter])
                            tempResult+=1
                        iter1 = wordIndex1

                if errorRes == False:
                    listOfCorrectParts3.append(newWord)
                    changes+=1
                    if tempResult > result:
                        result = tempResult
        listOfCorrectParts2 = listOfCorrectParts3

    return result
    pass
            

tab = ["abc","Banana","ghi","down"]

print(solution(tab))