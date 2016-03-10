###############################
##Created By: Bejan Sadeghian
##Date: 1/15/2016
##Purpose: Web Scraper Tutorial
###############################

from bs4 import BeautifulSoup
from urllib2 import urlopen
from time import sleep

def make_soup(url):
    html = urlopen(url).read()
    return BeautifulSoup(html, "lxml")

def get_category_links(section_url):
    soup = make_soup(section_url)
    boccat = soup.find("ul", "DataList Discussions")
    category_links = []
    for x in boccat:
        if x is not None:
            try:
                category_links.append(x.a['href'])
            except:
                continue
    return category_links

def get_post_information(category_url):
    soup= make_soup(category_url)
    #Original poster
    op_author = soup.find('div', 'MessageList Discussion').find('span','Author').find('a','Username').string
    op_date = soup.find('div', 'MessageList Discussion').find('span','MItem DateCreated').time['datetime']
    op_message = soup.find('div', 'MessageList Discussion').find('div','Message').string
    #Comments on post    
    comment_author_list = [comment.find('div','Comment') for comment in temp.find('div', 'CommentsWrap').find('ul','MessageList DataList Comments').find_all('li')]
    comment_author_list2 = []
    for x in comment_author_list:
        if x is not None:  #Sift through none elements
            comment_author_list2.append(x)
    comment_authors = [comment.find('a','Username').string for comment in comment_author_list2]
    comment_dates = [comment.time['datetime'] for comment in comment_author_list2]
    comment_messages = [comment.find('div','Message') for comment in comment_author_list2]
    comment_text = []
    for x in comment_messages:
        comment_munged = ''
        for y in x.contents:
            if str(y) != r'<br>' and str(y) != r'<br/>':
                comment_munged = comment_munged + ' ' + str(y).replace('\n', '')
        comment_text.append(comment_munged)
    
    return {"op_author": op_author,
            "op_date": op_date,
            "op_text": op_message,
            "comment_authors":comment_authors,
            "comment_dates":comment_dates,
            "comment_text":comment_text}

if __name__ == '__main__':
    best_of = r'http://forums.edmunds.com/discussions/tagged/x/general?Page=p1&'
    categories = get_category_links(best_of)
    
    list_of_posts = list()    
    for link in categories:
        list_of_posts.append(get_post_information(link))
        sleep(0.5) #To play nice
    print list_of_posts