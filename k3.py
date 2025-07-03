from thefuzz import fuzz
import pandas as pd

df = pd.read_csv("Data/hosts.csv")
spam_domains = set(df['host'].str.lower()) 

def is_spam_domain(text):
  
     for host in spam_domains:
        if fuzz.ratio(text, host) == 100:
            return True
     return False



name=[]


  
    
def detect(mail:str,t:list,server:str): 
 for i in t:
    name.append(i.split("@")[0])
 print(name)  
 if is_spam_domain(mail.split("@")[1])==True:
    print("Spam detected")
    return("spam domain")      
 elif (fuzz.ratio(server,mail.split("@")[1])>94 and fuzz.ratio(server,mail.split("@")[1])<100) :
    print("spam")
    return("spam almost similar domain")
 elif fuzz.ratio(server,mail.split("@")[1])<100:
   print("k")
   for i in name:
    mail1=mail.split("@")[0]
    print("mail1:", mail1)
    print("i:", i)
    print(f"Similarity score: {fuzz.ratio(mail1, i)}")
    if fuzz.ratio(mail1, i) > 50  :
         print("The names are similar.")
         return("spam similar names")
         exit(0)
    if fuzz.ratio(mail1,i) ==100:
            return("safe")
            exit(0)
            
           
 else:
        print("same server")
        return("safe")
        exit(0)
 return("safe")   
             
            
