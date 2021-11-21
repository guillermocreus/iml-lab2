# Using readlines()
file1 = open('EigenVectorCMC.txt', 'r')
Lines = file1.readlines()


 
count = 0
# Strips the newline character
for line in Lines:
    file2 = open('TableEigenVectorCMC.txt', 'a')
    x=line.split(' ')
    txtLine=""
    for i in x:
        try:
            txtLine=txtLine+str(float(i))+"&"
        except:
            print (i[:-1])
            txtLine=txtLine+str(float(i[:-1]))+"&"
    txtLine=txtLine+"\\\\\n"
    print (txtLine)
    file2.writelines(txtLine)
    file2.close()
	