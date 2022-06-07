# Нейроны сүлжээ ашиглан хүний үйл хөдлөлийг таних асуудалд
## Амарсайхан Дашням, Төмөрбаатар Батбилэг
Хүний байрлалыг тооцоолох нь хүний зан үйлийг таних, хөдөлгөөнийг
бүртгэх, бодит байдлыг нэмэгдүүлэх, роботуудыг сургах, хөдөлгөөн хянах
гэх мэт асар их ашиг тустай, хүний амьдралыг сайжруулах боломжит хэ-
рэглээ учраас сүүлийн үед компьютерын харааны асуудлын гол сэдэв болоод
байна. Deep Learning-тэй хэрэгжсэн орчин үеийн олон аргууд нь хэд хэдэн со-
рилтыг даван туулж, хүний байрлалыг тооцоолох салбарт гайхалтай үр дүнг
авчирсан. Арга барил нь хоёр үе шаттай (дээрээс доош чиглэсэн хандлага)
ба хэсэгчилсэн (доороос дээш хандлага) гэж хоёр ангилдаг. Хоёр үе шаттай
систем нь эхлээд хүн илрүүлэгчийг суулгаж, дараа нь хайрцаг тус бүрийн
байрлалыг бие даан тооцдог бол зураг дээрх биеийн бүх хэсгийг илрүүлж,
тодорхой хүмүүст хамаарах хэсгүүдийг холбох нь хэсэгчилсэн системд хийгд-
дэг. Энэхүү систем дээр суурилсан үйл хөдлөл таних аргачлалын судлан,
турших болно.Хиймэл оюун ухаан болон машин сургалт, нейрон сүлжээн та-
лаарх үндсэн ойлголт. Хүний биеийн бүтцийг тодорхойлох арга барилуудын
судалгаа хийж, Хүний үйл хөдлөлийг таних арга барилуудаас [Optical flow](https://www.facebook.com/batbileg.0724),
[хэт улаан туяаны гэрлийн камер](https://www.facebook.com/batbileg.0724), [хүний биеийн бүтцийн](https://www.facebook.com/batbileg.0724)гэх мэт олон
арга барил байдаг боловч аль нь үлүү болох нь эргэлзээтэй юм. Бүгд өөр
өөрийн өвөрмөц шийдэлтэй ба давуу болон сул талтай юм. Үзэгдэх орчин
хязгаарлагдмал, байгалын нөхцөл, шуугиан, эмх замбараагүй байдал гэх мэт
үзүүлэлтийг давуу талаараа нөхөж болдог ч сөрөг тал байсаар байна. Үүнд:
Техник, технологи, болон арга барилаас хамаарсан сул талууд юм. RGB бо-
лон Хүний биеийн бүтцийг сонгон авч өгөгдөл бэлдэн, харилцуулан дүгнэх
юм.


Орчин
-----

``` sh
$ python -m pip install -U pip
$ python -m pip install -U matplotlib
#=> matplotlib install

$ sudo apt install libopencv-dev python3-opencv
#=> cv2 install 

$ pip install mediapipe
#=> mediapipe install 

$ pip install tensorflow
#=> tensorflow install 
$ pip install keras
#=> keras install 
```
(Эхлэх)
-------------
```sh
$ git clone git@github.com:DashnyamDashaa/Dashnyam-Batbileg.git
$ cd Dashnyam-Batbileg
```
(Өгөгдөл боловсруулах)
-------
### [Өгөгдөл боловсруулахын өмнө](https://github.com/DashnyamDashaa/Dashnyam-Batbileg/blob/master/data/readme.md) өгөгдөл бэлдсэн байх шаардлагатай юм.
```sh
$ mkdir testdata/out $$ testdata/img
$ python main.py -data
```
(Дахин сургах)
-------
### [Дахин сургахын өмнө](https://github.com/DashnyamDashaa/Dashnyam-Batbileg/blob/master/testdata/readme.md) өгөгдөл бэлдсэн байх шаардлагатай юм. Мөн сургах загвараа сонгохдоо `sict`, `vgg16`, `simple` зэргээс сонгон дүрслэнэ. Дахин сургалт дууссны дараа [хадгалагдна](https://github.com/DashnyamDashaa/Dashnyam-Batbileg/blob/master/modelH5/readme.md).
```sh
$ python main.py -data sict
```
(Үндсэн ажиллаггаа)
-------
### Үндсэн ажиллаггааны өмнө болон [Дахин сургах](https://github.com/DashnyamDashaa/Dashnyam-Batbileg/blob/master/testdata/readme.md) болон [загвар](https://github.com/DashnyamDashaa/Dashnyam-Batbileg/blob/master/modelH5/readme.md) үүссэн байх шаардлагатай юм. Мөн сургах загвараа сонгохдоо `sict`, `vgg16`, `simple` зэргээс сонгон дүрслэнэ. Дахин сургалт дууссны дараа [хадгалагдна](https://github.com/DashnyamDashaa/Dashnyam-Batbileg/blob/master/modelH5/readme.md).
```sh
$ python main.py -a sict
```

<!-- Installation -->
<!-- ------------ -->
<!-- The `hub` executable has no dependencies, but since it was designed to wrap -->
<!-- `git`, it's recommended to have at least **git 1.7.3** or newer. -->
<!-- platform | manager | command to run -->
<!-- ---------|---------|--------------- -->
<!-- macOS, Linux | [Homebrew](https://docs.brew.sh/Installation) | `brew install hub` -->