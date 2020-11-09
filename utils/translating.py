from googletrans import Translator

class GoogleTranslator:
    def __init__(self, src='en', dest='vi'):
        self.translator = Translator()
        self.src = src
        self.dest = dest
        
    def detect(self, text):
        while True:
            try:
                lang = self.translator.detect(text)
                break
            except Exception as e:
                self.translator = Translator()
        return lang.lang, lang.confidence
        
    def translate(self, text, patience_lim=7):
        patience = 0
        while True:
            try:
                result = self.translator.translate(text,
                                                   src=self.src,
                                                   dest=self.dest,
                                                   service_urls=['translate.google.com']
                                                   ).text
                if result != text:
                    break
                else:
                    patience += 1
                    if (patience >= patience_lim):
                        return None
            except Exception as e:
                self.translator = Translator()
        return result