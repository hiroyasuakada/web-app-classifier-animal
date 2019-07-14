from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

# API key Information

key = "5b466227d57ec99f08968bdc7dd21a2b"
secret = "3cd93ffa069e85c1"
wait_time = 1

# to designate save folder
animalname = sys.argv[1]
savedir = "./" + animalname

flickr = FlickrAPI(key, secret, format = 'parsed-json')
result = flickr.photos.search(
    text = animalname,
    per_page = 400,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, licence',
)

photos = result['photos']

# tp display a return value
# pprint(photos)

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'

    # to confirm there is no duplicate
    if os.path.exists(filepath): continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)
