
import argparse
from driver import FINNAccelDriver
import numpy as np

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Validate top-1 accuracy for FINN accelerator')
  parser.add_argument('--batchsize', help='number of samples for inference', type=int, default=100)
  parser.add_argument('--dataset', help='dataset to use (mnist of cifar10)', required=True)
  # parse arguments
  args = parser.parse_args()
  bsize = args.batchsize
  dataset = args.dataset

  if dataset == "mnist":
    from dataset_loading import mnist
    trainx, trainy, testx, testy, valx, valy = mnist.load_mnist_data("/tmp", download=True, one_hot=False)
  elif dataset == "cifar10":
    from dataset_loading import cifar
    trainx, trainy, testx, testy, valx, valy = cifar.load_cifar_data("/tmp", download=True, one_hot=False)
  else:
    raise Exception("Unrecognized dataset")

  test_imgs = testx
  test_labels = testy

  ok = 0
  nok = 0
  total = test_imgs.shape[0]
  driver = FINNAccelDriver(bsize, "resizer.bit", "zynq-iodma")

  n_batches = int(total / bsize)

  test_imgs = test_imgs.reshape(n_batches, bsize, -1)
  test_labels = test_labels.reshape(n_batches, bsize)

  for i in range(n_batches):
    ibuf_normal = test_imgs[i].reshape(driver.ibuf_packed_device.shape)
    exp = test_labels[i]
    driver.copy_input_data_to_device(ibuf_normal)
    driver.execute()
    obuf_normal = np.empty_like(driver.obuf_packed_device)
    driver.copy_output_data_from_device(obuf_normal)
    ret = np.bincount(obuf_normal.flatten() == exp.flatten())
    nok += ret[0]
    ok += ret[1]
    print("batch %d / %d : total OK %d NOK %d" % (i, n_batches, ok, nok))

  acc = 100.0 * ok / (total)
  print("Final accuracy: %f" % acc)
