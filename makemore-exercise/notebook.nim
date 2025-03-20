import 
  std/[os, strutils, sequtils, tables, algorithm, math],
  ./makemore, ../nim_micrograd/src/nim_micrograd/[nn, engine]

# https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2
# https://karpathy.ai/zero-to-hero.html
# "building makemore"

proc main() =

  let words = readFile("names.txt").splitLines()
  echo "words: ", words.len

  # get the shortest word
  let shortestWord = words.mapIt(it.len).min()
  echo "shortest word: ", shortestWord

  # get the longest word
  let longestWord = words.mapIt(it.len).max()
  echo "longest word: ", longestWord

  # build a bigram
  var b = initTable[(string, string), int]()
  for w in words:
    let wordChars = toSeq(w.items).map(proc (c: char): string = $c)
    var chs = @["<S>"] 
    chs.add(wordChars) 
    chs.add("<E>") 
    for i in 0 ..< chs.len - 1:
      let bigram = (chs[i], chs[i + 1])
      b[bigram] = b.getOrDefault(bigram) + 1

  block:
    ## Print the top 10 bigrams
    # Convert table to sequence of (key, value) pairs and sort
    let sortedBigrams = toSeq(b.pairs).sorted(proc (a, b: ((string, string), int)): int =
      # Sort descending by value (count), hence b[1] - a[1]
      cmp(b[1], a[1])
    )

    for i in 0 .. min(9, sortedBigrams.len - 1):
      let (bigram, count) = sortedBigrams[i]
      echo "(", bigram[0], ", ", bigram[1], "): ", count


  # TODO next step
  # make a torch tensor of 27,27 size of int32 with all zeros
  # but instead use nim_micrograd

when isMainModule:
  main()
