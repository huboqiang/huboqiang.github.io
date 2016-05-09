---
title: "Use multiple CPU Cores with your Linux commands — awk, sed, bzip2, grep, wc, etc."
tagline: ""
last_updated: 2016-05-09
category: tools
layout: post
tags : [parallel, multiple CPU]
---

# Use multiple CPU Cores with your Linux commands — awk, sed, bzip2, grep, wc, etc.

This article were copied from [Aris's blog](http://www.rankfocus.com/use-cpu-cores-linux-commands/)

Here’s a  common problem: You ever want to add up a very large list (hundreds of megabytes) or grep through it, or other kind of operation that is embarrassingly parallel? Data scientists, I am talking to you. You probably  have about four cores or more, but our tried and true tools like grep, bzip2, wc, awk, sed and so forth are singly-threaded and will just use one CPU core. To paraphrase Cartman, “How do I reach these cores”? Let’s use all of our CPU cores on our Linux box by using GNU Parallel and doing a little in-machine map-reduce magic by using all of our cores and using the little-known parameter –pipes (otherwise known as –spreadstdin). Your pleasure is proportional to the number of CPUs, I promise.   BZIP2 So, bzip2 is better compression than gzip, but it’s so slow! Put down the razor, we have the technology to solve this. Instead of this:

```
cat bigfile.bin | bzip2 --best > compressedfile.bz2
```

Do this:

```
cat bigfile.bin | parallel --pipe --recend '' -k bzip2 --best > compressedfile.bz2
```

Especially with bzip2, GNU parallel is dramatically faster on multiple core machines. Give it a whirl and you will be sold.     GREP If you have an enormous text file, rather than this:

grep pattern bigfile.txt
do this:

```
cat bigfile.txt | parallel  --pipe grep 'pattern'
```

or this:

```
cat bigfile.txt | parallel --block 10M --pipe grep 'pattern'
```

These second command shows you using –block with 10 MB of data from your file — you might play with this parameter to find our how many input record lines you want per CPU core. I gave a previous example of how to use grep with a large number of files, rather than just a single large file. AWK Here’s an example of using awk to add up the numbers in a very large file. Rather than this:

```
cat rands20M.txt | awk '{s+=$1} END {print s}'
```

do this!

```
cat rands20M.txt | parallel --pipe awk \'{s+=\$1} END {print s}\' | awk '{s+=$1} END {print s}'
```

This is more involved: the –pipe option in parallel spreads out the output to multiple chunks for the awk call, giving a bunch of sub-totals. These sub totals go into the second pipe with the identical awk call, which gives the final total. The first awk call has three backslashes in there due to the need to escape the awk call for GNU parallel. WC Want to create a super-parallel count of lines in a file? Instead of this:

```
wc -l bigfile.txt
```

Do this:

```
cat bigfile.txt | parallel  --pipe wc -l | awk '{s+=$1} END {print s}'
```

This is pretty neat: What is happening here is during the parallel call, we are ‘mapping’ a bunch of calls to wc -l , generating sub-totals, and finally adding them up with the final pipe pointing to awk. SED Feel like using sed to do a huge number of replacements in a huge file? Instead of this:

```
sed s^old^new^g bigfile.txt
```

Do this:

```
cat bigfile.txt | parallel --pipe sed s^old^new^g
```

…and then pipe it into your favorite file to store the output.

Enjoy!
