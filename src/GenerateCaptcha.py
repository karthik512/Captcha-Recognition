from captcha.image import ImageCaptcha
import os
from random import *


allowedFontType = ['.ttf', '.woff2', '.svg', '.eot','.otf']
fontFiles = []
fontDirPath = os.getcwd() + '\..\\fonts'
dirList = os.listdir(fontDirPath)
for item in dirList:
    for fontType in allowedFontType:
        if fontType in item:
            fontFiles.append(fontDirPath + '\\' + item)

allowedChars = "1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TOTAL_IMAGES = 5

for captchaLen in range(6,7):
    for noSamples in range(TOTAL_IMAGES):
        if noSamples % 500 == 0:
            noFonts = randint(1, len(fontFiles))
            selectedFonts = sample(fontFiles, noFonts)

        captchaDirName = '..\\data\\TestCode\\'
        selectedCaptcha = ''.join(sample(allowedChars, captchaLen))

        image = ImageCaptcha(fonts=selectedFonts)
        data = image.generate(selectedCaptcha)
        image.write(selectedCaptcha, captchaDirName + selectedCaptcha + '.png')
    print("Completed Generating for CaptchaLength : {0}".format(captchaLen))