import urllib.request, re
# from bs4 import BeautifulSoup
import wikipedia as wiki

# def connect_to_page(word, recur):
#     try:
#         url = 'https://en.wikipedia.org/wiki/'+word
#         req = urllib.request.Request(url, headers={'User-Agent' : "Mozilla/5.0 (Windows NT 6.0; WOW64; rv:12.0) Gecko/20100101 Firefox/12.0"}) 
#         page = urllib.request.urlopen(req).read().decode('utf-8')

#         if recur == False:
#             if '</b> may refer to:</p>\n<ul>' in page:
#                 possible_meanings = re.findall('<li>[a-zA-Z ,\-]*<a href="/wiki/([^"]+)"', page)
#                 page = connect_to_page(possible_meanings[0], True)
                      
#     except urllib.error.HTTPError:
#         page = ''

#     except urllib.error.URLError:
#         time.sleep(5)
#         page = connect_to_page(word, False)
#     return(page)

testing = open('validation.txt', 'r', encoding='utf-8')
features = open('validation_wiki.txt', 'w', encoding='utf-8')

for line in testing:
    entries = line.strip('\n').split(',')
    try:
    	occurences_1 = len(re.findall(entries[2],
    		wiki.page(entries[0]).content))
    except Exception as e:
    	print(e)
    	occurences_1 = 0

    try:
    	occurences_2 = len(re.findall(entries[2],
    		wiki.page(entries[1]).content))
    except Exception as e:
    	print(e)
    	occurences_2 = 0

    # page = connect_to_page(entries[0].capitalize(), False)
    # occurences_1 = len(re.findall(entries[2], page))
    # page = connect_to_page(entries[1].capitalize(), False)
    # occurences_2 = len(re.findall(entries[2], page))

    features.write(entries[0]+','+entries[1]+','+entries[2]+','+entries[3]+','+str(occurences_1)+','+str(occurences_2)+'\n')
features.close()

    
