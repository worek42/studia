import random



DNALettersDict = {0: "A", 1: "T", 2: "C", 3: "G"}

#===============================================================================================
#Generate DNA sequence for the test
#===============================================================================================
OriginalDNALength = 1000

OriginalDNASeq = [DNALettersDict[random.randint(0, 3)] for i in range(OriginalDNALength)]

#print(OriginalDNASeq)

#===============================================================================================
#Generate Fragments
#===============================================================================================
NumberOfFragments = 10000
MinLength = 50
MaxLength = 400

def SelFragment(DNASeq):
  startPos = random.randint(0, OriginalDNALength-1-MinLength)
  length = random.randint(MinLength, MaxLength)
  return DNASeq[startPos:startPos+length]

ListOfFragments = [ SelFragment(OriginalDNASeq) for i in range(NumberOfFragments)]


is_Not_Done = True

while is_Not_Done :
  is_Not_Done = False
  le = len(ListOfFragments)

  for i in range(0,le) :
    if(i >= le):
      break

    reference = str(ListOfFragments[i])[1:-1]

    for j in range(0,le) :
      test = str(ListOfFragments[j])[1:-1]

      if (test in reference) and (i != j):
        ListOfFragments.remove(ListOfFragments[j])
        le -= 1
        is_Not_Done = True
        break

le = len(ListOfFragments)

while le > 1 :
  le = len(ListOfFragments)

  for i in range(0,le) :
    if(i >= le):
      break

    reference = str(ListOfFragments[i])[1:-1]

    for j in range(0,le) :
      test1 = str(ListOfFragments[j])[1:-1]
      test2 = []
      if (test1 in reference) and (i != j):

        le -= 1
        break

print("cos")

        






#===============================================================================================
#Reconstruct DNA from fragments
#===============================================================================================

