---
layout: post
title:  "Vim Editor 참고 사항들"
date:   2020-12-04
categories:
    - "K-Digital Training"
    - "Linux"
---

## vi (visual editor) vs. vim (vim improved)
- vi
    - UNIX / Linux에서 가장 많이 사용하는 에디터
    - 1976년 BSD의 Bill Joy가 개발
    - 지금은 쓰이지 않음
- vim
    - vi에 추가적인 확장 기능을 부여한 에디터
    - 다양한 플랫폼 지원 (Linux, UNIX, Mac OSX, Windows)


-----


## vim 설치
- RedHat 계열: RHEL, CentOS, Fedora
{% highlight shell-session %}
# yum -y install vim-enhanced
{% endhighlight %}
- Debian 계열: Debian, Ubuntu, Mint...
{% highlight shell-session %}
$ yum -y install vim-enhanced
{% endhighlight %}


-----


## vim [filename]
- 특정 파일명을 열면서 시작
- `$ vim mytext.txt`
- `$ find . -name “*.txt” | vim -`
    - 현재 디렉토리에서 “*.txt”를 찾은 결과를 화면에 출력할 결과를 vim에 저장하라.
    - `vim -`: **`-` 의미는 stdin에 있는 것을 읽어오라는 뜻**


-----


## vi[m]의 기본 작동 모드

1. 일반 모드 (normal mode)
2. 입력 모드 (insert mode)
3. 명령행 모드 (command-line mode)
4. [비주얼 모드 (visual mode)] vim에서 추가된 모드

