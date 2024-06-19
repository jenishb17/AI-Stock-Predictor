from retrain_scheduler import RetrainScheduler
from app import app

if __name__ == '__main__':
    retrain_scheduler = RetrainScheduler()
    retrain_scheduler.start()

    app.run(debug=True)
