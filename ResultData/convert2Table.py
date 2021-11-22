# Using readlines()
file1 = open('ConHypo.txt', 'r')
Lines = file1.readlines()


 
count = 0
n=3
# Strips the newline character
for line in Lines:
    file2 = open('TableConHypo.txt', 'a')
    x=line.split(' ')
    txtLine=""
    for i in x:
        try:
            txtLine=txtLine+str( ("{0:.%ie}" % (n-1)).format(float(i)))+"&"
        except:
            print (i[:-1])
            txtLine=txtLine+str(("{0:.%ie}" % (n-1)).format(float(i[:-1])))
    txtLine=txtLine+"\\\\\n"
    print (txtLine)
    file2.writelines(txtLine)
    file2.close()