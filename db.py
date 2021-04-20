import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# setup db code
Base = declarative_base()

# create table as python class
class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer,primary_key=True)
    path = Column(String)
    uploaded_on = Column(DateTime,default=datetime.now)

    def __str__(self):
        return self.path
    
    def __repr__(self) -> str:
        return self.path

class Video(Base):
    __tablename__ = 'video'
    id = Column(Integer,primary_key=True)
    path = Column(String)
    uploaded_on = Column(DateTime,default=datetime.now)

    def __str__(self):
        return {self.path}
    
    def __repr__(self) -> str:
        return self.path

# create database
if __name__ == "__main__":
    engine = create_engine("sqlite:///database/db.sqlite3")
    Base.metadata.create_all(engine)