from sqlalchemy import Column, DateTime, Float, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Sensors(Base):
    __tablename__ = 'sensors'

    date = Column(DateTime, primary_key=True)
    p1 = Column(Float)
    p2 = Column(Float)
    temperature = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)


class Weather(Base):
    __tablename__ = "weather"

    date = Column(DateTime, primary_key=True)
    temp = Column(Float)
    press = Column(Float)
    prec = Column(String)
    wind_speed = Column(Float)
    wind_dir = Column(String)
    hum = Column(Float)