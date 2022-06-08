# Нейроны сүлжээ ашиглан хүний үйл хөдлөлийг таних асуудалд
## Амарсайхан Дашням, Төмөрбаатар Батбилэг
Хүний байрлалыг тооцоолох нь хүний зан үйлийг таних, хөдөлгөөнийг бүртгэх, бодит байдлыг нэмэгдүүлэх, роботуудыг сургах, хөдөлгөөн хянах гэх мэт асар их ашиг тустай, хүний амьдралыг сайжруулах боломжит хэрэглээ учраас сүүлийн үед компьютерын харааны асуудлын гол сэдэв болоод байна. Deep Learning-тэй хэрэгжсэн орчин үеийн олон аргууд нь хэд эдэн сорилтыг даван туулж, хүний байрлалыг тооцоолох салбарт гайхалтай үр дүнг авчирсан. Арга барил нь хоёр үе шаттай (дээрээс доош чиглэсэн хандлага) ба хэсэгчилсэн (доороос дээш хандлага) гэж хоёр ангилдаг. Хоёр үе шаттай систем нь эхлээд хүн илрүүлэгчийг суулгаж, дараа нь хайрцаг тус бүрийн байрлалыг бие даан тооцдог бол зураг дээрх биеийн бүх хэсгийг илрүүлж, тодорхой хүмүүст хамаарах хэсгүүдийг холбох нь хэсэгчилсэн системд хийгддэг. Энэхүү систем дээр суурилсан үйл хөдлөл таних аргачлалын судлан, турших болно.Хиймэл оюун ухаан болон машин сургалт, нейрон сүлжээн талаарх үндсэн ойлголт. Хүний биеийн бүтцийг тодорхойлох арга барилуудын судалгаа хийж, Хүний үйл хөдлөлийг таних арга барилуудаас [Optical flow](https://en.wikipedia.org/wiki/Optical_flow), [хэт улаан туяаны гэрлийн камер](https://www.researchgate.net/publication/274091798_Human_Detection_Based_on_the_Generation_of_a_Background_Image_by_Using_a_Far-Infrared_Light_Camera), [хүний биеийн бүтцийн](https://arxiv.org/abs/2104.11712?fbclid=IwAR2SwqSOHi3if7cFbnEd6QOMFZw_1StPRkvL7WFktVRWG1afYWZmmBTz2l4#:~:text=Skeletor%3A%20Skeletal%20Transformers%20for%20Robust%20Body%2DPose%20Estimation,-Tao%20Jiang%2C%20Necati&text=Predicting%203D%20human%20pose%20from,in%20estimating%203D%20from%202D)гэх мэт олон арга барил байдаг боловч аль нь илүү болох нь эргэлзээтэй юм. Бүгд өөр өөрийн өвөрмөц шийдэлтэй ба давуу болон сул талтай юм. Үзэгдэх орчин хязгаарлагдмал, байгалын нөхцөл, шуугиан, эмх замбараагүй байдал гэх мэт үзүүлэлтийг давуу талаараа нөхөж болдог ч сөрөг тал байсаар байна. Үүнд: Техник, технологи, болон арга барилаас хамаарсан сул талууд юм. RGB болон Хүний биеийн бүтцийг сонгон авч өгөгдөл бэлдэн, харилцуулан дүгнэх юм.


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
### [Дахин сургахын өмнө](https://github.com/DashnyamDashaa/Dashnyam-Batbileg/blob/master/testdata/readme.md) өгөгдөл бэлдсэн байх шаардлагатай юм. Мөн сургах загвараа сонгохдоо `sict`, `vgg16`, `simple` зэргээс сонгон дүрслэнэ. Дахин сургалт дууссаны дараа [хадгалагдана](https://github.com/DashnyamDashaa/Dashnyam-Batbileg/blob/master/modelH5/readme.md).
```sh
$ python main.py -data sict
```
(Үндсэн ажиллаггаа)
-------
### Үндсэн ажиллагааны өмнө болон [Дахин сургах](https://github.com/DashnyamDashaa/Dashnyam-Batbileg/blob/master/testdata/readme.md) болон [загвар](https://github.com/DashnyamDashaa/Dashnyam-Batbileg/blob/master/modelH5/readme.md) үүссэн байх шаардлагатай юм. Мөн сургах загвараа сонгохдоо `sict`, `vgg16`, `simple` зэргээс сонгон дүрсэлнэ. Дахин сургалт дууссаны дараа [хадгалагдана](https://github.com/DashnyamDashaa/Dashnyam-Batbileg/blob/master/modelH5/readme.md).
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