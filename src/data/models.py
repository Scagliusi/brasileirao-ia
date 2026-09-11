from sqlalchemy import (
    Column,
    Integer,
    String,
    Date,
    ForeignKey,
)

from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base() #cria a arquitetura do banco de dados, para que seja possível criar tabelas e relacionamentos entre elas.

# Todas as Classes criadas com (base) SQLAlchemy interpretará como tabelas no banco de dados, e cada atributo da classe será uma coluna na tabela correspondente. exceto o relationship, que é usado para criar relacionamentos entre tabelas, mas não cria uma coluna no banco de dados.

class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    short_name = Column(String)


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)

    season = Column(Integer, nullable=False)
    round = Column(Integer)
    date = Column(Date)

    home_team_id = Column(Integer, ForeignKey("teams.id")) # cria conexão com a tabela de times, para que seja possível fazer consultas mais complexas no banco de dados.
    away_team_id = Column(Integer, ForeignKey("teams.id"))

    home_goals = Column(Integer)
    away_goals = Column(Integer)

    status = Column(String)

    home_team = relationship(  #Facilita na coleta de dados com o python. Exp: Ao receber o ID 3, pode retornar direto o nome do time associado a esse ID, sem precisar fazer outra query no banco de dados.
        "Team",
        foreign_keys=[home_team_id]
    )

    away_team = relationship(
        "Team",
        foreign_keys=[away_team_id]
    )