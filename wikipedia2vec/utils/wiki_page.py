import re
from typing import Union

import cython

# Obtained from https://github.com/spencermountain/wtf_wikipedia/blob/32d6737499a882141c2875c4993156793ba31fe0/builds/wtf_wikipedia.mjs#L466
# Short template names (z and db) are commented out.
DISAMBIGUATION_TEMPLATE_NAMES = frozenset(
    [
        "dab",  # en
        "disamb",  # en
        "disambig",  # en
        "disambiguation",  # en
        "aðgreining",  # is
        "aimai",  # ja
        "airport disambiguation",
        "ałtsʼáʼáztiin",  # nv
        "anlam ayrımı",  # gag
        "anlam ayrımı",  # tr
        "apartigilo",  # eo
        "argipen",  # eu
        "begriepskloorenge",  # stq
        "begriffsklärung",  # als
        "begriffsklärung",  # de
        "begriffsklärung",  # pdc
        "begriffsklearung",  # bar
        "biology disambiguation",
        "bisongidila",  # kg
        "bkl",  # pfl
        "bokokani",  # ln
        "caddayn",  # so
        "call sign disambiguation",
        "caselaw disambiguation",
        "chinese title disambiguation",
        "clerheans",  # kw
        "cudakirin",  # ku
        "čvor",  # bs
        # 'db', # vls
        "desambig",  # nov
        "desambigación",  # an
        "desambiguação",  # pt
        "desambiguació",  # ca
        "desambiguación",  # es
        "desambiguáncia",  # ext
        "desambiguasion",  # lad
        "desambiguassiù",  # lmo
        "desambigui",  # lfn
        "dezambiguizare",  # ro
        "dezanbìgua",
        "dəqiqləşdirmə",  # az
        "disamb-term",
        "disamb-terms",
        "disamb2",
        "disamb3",
        "disamb4",
        "disambigua",  # it
        "disambìgua",  # sc
        "disambiguasi",
        "disambiguation cleanup",
        "disambiguation lead name",
        "disambiguation lead",
        "disambiguation name",
        "disambiguazion",
        "disambigue",
        "discretiva",  # la
        "disheñvelout",  # br
        "disingkek",  # min
        "dixanbigua",  # vec
        "dixebra",  # ast
        "diżambigwazzjoni",  # mt
        "dmbox",
        "doorverwijspagina",  # nl
        "dubbelsinnig",  # af
        "dudalipen",  # rmy
        "egyért",  # hu
        "faaleaogaina",
        "fleiri týdningar",  # fo
        "fleirtyding",  # nn
        "flertydig",  # da
        "förgrening",  # sv
        "genus disambiguation",
        "gì-ngiê",  # cdo
        "giklaro",  # ceb
        "gwahaniaethu",  # cy
        "homonimo",  # io
        "homónimos",  # gl
        "homonymie",  # fr
        "hospital disambiguation",
        "huaʻōlelo puana like",  # haw
        "human name disambiguation cleanup",
        "human name disambiguation",
        "idirdhealú",  # ga
        "khu-pia̍t",  # zh_min_nan
        "kthjellim",  # sq
        "kujekesa",  # sn
        "letter-number combination disambiguation",
        "letter-numbercombdisambig",
        "maana",  # sw
        "maneo bin",  # diq
        "mathematical disambiguation",
        "mehrdüdig begreep",  # nds
        "menm non",  # ht
        "military unit disambiguation",
        "muardüüdag artiikel",  # frr
        "music disambiguation",
        "myesakãrã",
        "neibetsjuttings",  # fy
        "nozīmju atdalīšana",  # lv
        "number disambiguation",
        "nuorodinis",  # lt
        "nyahkekaburan",  # ms
        "omonimeye",  # wa
        "omonimi",
        "omonimia",  # oc
        "opus number disambiguation",
        "page dé frouque",  # nrm
        "paglilinaw",  # tl
        "panangilawlawag",  # ilo
        "pansayod",  # war
        "pejy mitovy anarana",  # mg
        "peker",  # no
        "phonetics disambiguation",
        "place name disambiguation",
        "portal disambiguation",
        "razdvojba",  # hr
        "razločitev",  # sl
        "razvrstavanje",  # sh
        "reddaghey",  # gv
        "road disambiguation",
        "rozcestník",  # cs
        "rozlišovacia stránka",  # sk
        "school disambiguation",
        "sclerir noziun",  # rm
        "selvendyssivu",  # olo
        "soilleireachadh",  # gd
        "species latin name abbreviation disambiguation",
        "species latin name disambiguation",
        "station disambiguation",
        "suzmunski",  # jbo
        "synagogue disambiguation",
        "täpsustuslehekülg",  # et
        "täsmennyssivu",  # fi
        "taxonomic authority disambiguation",
        "taxonomy disambiguation",
        "telplänov",  # vo
        "template disambiguation",
        "tlahtolmelahuacatlaliztli",  # nah
        "trang định hướng",  # vi
        "ujednoznacznienie",  # pl
        "verdudeliking",  # li
        "wěcejwóznamowosć",  # dsb
        "wjacezmyslnosć",  # hsb
        # "z",
        "zambiguaçon",  # mwl
        "zeimeibu škiršona",  # ltg
        "αποσαφήνιση",  # el
        "айрық",  # kk
        "аҵакырацәа",  # ab
        "бир аайы јок",
        "вишезначна одредница",  # sr
        "ибҳомзудоӣ",  # tg
        "кёб магъаналы",  # krc
        "күп мәгънәләр",  # tt
        "күп мәғәнәлелек",  # ba
        "массехк маӏан хилар",
        "мъногосъмꙑслиѥ",  # cu
        "неадназначнасць",  # be
        "неадназначнасьць",  # be_x_old
        "неоднозначность",  # ru
        "олон удхатай",  # bxr
        "појаснување",  # mk
        "пояснение",  # bg
        "са шумуд манавал",  # lez
        "салаа утгатай",  # mn
        "суолталар",  # sah
        "текмаанисиздик",  # ky
        "цо магіна гуреб",  # av
        "чеперушка",  # rue
        "чолхалла",  # ce
        "шуко ончыктымаш-влак",  # mhr
        "მრავალმნიშვნელოვანი",  # ka
        "բազմիմաստութիւն",  # hyw
        "բազմիմաստություն",  # hy
        "באדייטן",  # yi
        "פירושונים",  # he
        "ابهام‌زدایی",  # fa
        "توضيح",  # ar
        "توضيح",  # arz
        "دقیقلشدیرمه",  # azb
        "ڕوونکردنەوە",  # ckb
        "سلجهائپ",  # sd
        "ضد ابہام",  # ur
        "گجگجی بیری",  # mzn
        "نامبهمېدنه",  # ps
        "መንታ",  # am
        "अस्पष्टता",  # ne
        "बहुअर्थी",  # bh
        "बहुविकल्पी शब्द",  # hi
        "দ্ব্যর্থতা নিরসন",  # bn
        "ਗੁੰਝਲ-ਖੋਲ੍ਹ",  # pa
        "સંદિગ્ધ શીર્ષક",  # gu
        "பக்கவழி நெறிப்படுத்தல்",  # ta
        "అయోమయ నివృత్తి",  # te
        "ದ್ವಂದ್ವ ನಿವಾರಣೆ",  # kn
        "വിവക്ഷകൾ",  # ml
        "වක්‍රෝත්ති",  # si
        "แก้ความกำกวม",  # th
        "သံတူကြောင်းကွဲ",  # my
        "သဵင်မိူၼ် တူၼ်ႈထႅဝ်ပႅၵ်ႇ",
        "ណែនាំ",  # km
        "អសង្ស័យកម្ម",
        "동음이의",  # ko
        "扤清楚",  # gan
        "搞清楚",  # zh_yue
        "曖昧さ回避",  # ja
        "消歧义",  # zh
        "釋義",  # zh_classical
        "gestion dj'omònim",  # pms
        "sut'ichana qillqa",  # qu
    ]
)
TEMPLATE_REGEXP = re.compile(r"{{\s*([^}\|]+?)\s*(?:\||})")


@cython.cclass
class WikiPage:
    title = cython.declare(str, visibility="readonly")
    language = cython.declare(str, visibility="readonly")
    wiki_text = cython.declare(str, visibility="readonly")
    redirect = cython.declare(str, visibility="readonly")

    def __init__(self, title: str, language: str, wiki_text: str, redirect: Union[str, None]):
        self.title = title
        self.language = language
        self.wiki_text = wiki_text
        self.redirect = redirect

    @property
    def is_redirect(self) -> bool:
        return bool(self.redirect)

    @property
    def is_disambiguation(self) -> bool:
        return any(name.lower() in DISAMBIGUATION_TEMPLATE_NAMES for name in TEMPLATE_REGEXP.findall(self.wiki_text))

    def __repr__(self):
        return f"<WikiPage {self.title}>"
