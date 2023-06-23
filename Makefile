run_train:
	./train.sh

open_tensorboard:
	tensorboard --logdir textual_inversion_target/logs --bind_all

generate_skater:
	python generate "A <target> as a skater"

generate_car:
	python generate "A <target> as a silver toy car"