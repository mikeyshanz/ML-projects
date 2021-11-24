from google_images_search import GoogleImagesSearch

# if you don't enter api key and cx, the package will try to search
# them from environment variables GCS_DEVELOPER_KEY and GCS_CX

# AIzaSyB3_c5Z_BSk6SWKwzPjQZGH72dUUiKZl8Y
# 000305825002348842648:txhdedsiczo

gis = GoogleImagesSearch('AIzaSyB3_c5Z_BSk6SWKwzPjQZGH72dUUiKZl8Y', '000305825002348842648:txhdedsiczo')

# example: GoogleImagesSearch('ABcDeFGhiJKLmnopqweRty5asdfghGfdSaS4abC', '012345678987654321012:abcde_fghij')

# define search params:
_search_params = {
    'q': '...',
    'num': 1-50,
    'safe': 'off',
    'fileType': 'jpg',
    'imgType': 'face',
    'imgSize': 'huge|icon|large|medium|small|xlarge|xxlarge',
    'searchType': 'image',
    'imgDominantColor': 'black|blue|brown|gray|green|pink|purple|teal|white|yellow'
}

# this will only search for images:
gis.search(search_params=_search_params)

# this will search and download:
gis.search(search_params=_search_params, path_to_dir='/path/')

# this will search, download and resize:
gis.search(search_params=_search_params, path_to_dir='/path/', width=500, height=500)

# search first, then download and resize afterwards
gis.search(search_params=_search_params)
for image in gis.results():
    image.download('/path/')
    image.resize(500, 500)
