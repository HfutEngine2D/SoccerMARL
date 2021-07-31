killall -9 rcssserver
killall -9 ray::PPO.train_buffered\(\)
killall -9 ray::MARWIL.train_buffered\(\)
killall ray::RolloutWorker.sample\(\)
killall ray::RolloutWorker.par_iter_next\(\)