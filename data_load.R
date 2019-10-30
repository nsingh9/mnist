download.file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
              "train-images-idx3-ubyte.gz")
download.file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
              "train-labels-idx1-ubyte.gz")
download.file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
              "t10k-images-idx3-ubyte.gz")
download.file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
              "t10k-labels-idx1-ubyte.gz")

# gunzip the files
R.utils::gunzip("train-images.idx3-ubyte.gz")
R.utils::gunzip("train-labels.idx1-ubyte.gz")
R.utils::gunzip("t10k-images.idx3-ubyte.gz")
R.utils::gunzip("t10k-labels.idx1-ubyte.gz")

# helper function for visualization
show_digit = function(arr784, col = gray(12:1 / 12), ...) {
  image(matrix(as.matrix(arr784[-785]), nrow = 28)[, 28:1], col = col, ...)
}

# load image files
load_image_file <- function(filename) {
  ret = list()
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=1,endian='big')
  ret$n = readBin(f,'integer',n=1,size=1,endian='big')
  nrow = readBin(f,'integer',n=1,size=1,endian='big')
  ncol = readBin(f,'integer',n=1,size=1,endian='big')
  x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
  ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
  close(f)
  ret
}

# load label files
load_label_file <- function(filename) {
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  n = readBin(f,'integer',n=1,size=4,endian='big')
  y = readBin(f,'integer',n=n,size=1,signed=F)
  close(f)
  y
}

# load images
train = load_image_file("train-images.idx3-ubyte")
test  = load_image_file("t10k-images.idx3-ubyte")

# load labels
train$y = as.factor(load_label_file("train-labels.idx1-ubyte"))
test$y  = as.factor(load_label_file("t10k-labels.idx1-ubyte"))

# view test image
show_digit(train[10000, ])

# testing classification on subset of training data
fit = randomForest::randomForest(y ~ ., data = train[1:1000, ])
fit$confusion
test_pred = predict(fit, test)
mean(test_pred == test$y)
table(predicted = test_pred, actual = test$y)