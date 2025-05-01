import yt_dlp as ydl

# Open the links file
with open("links.txt", "r") as theLinks:
    links = theLinks.read().split("\n")

# Configuring ydl options
ydl_opts = {
    'format': 'bestvideo[height<=720]+bestaudio/best',
    'outtmpl': '/Users/jasper/Desktop/ATFinal/AT-Final-Project/Dataset/Videos/%(title)s.%(ext)s'
}

# download each video
with ydl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(links)
