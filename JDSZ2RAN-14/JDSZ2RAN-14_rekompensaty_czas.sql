rok
select extract(year from data_utworzenia)as lata,avg(kwota_rekompensaty)as srednia,sum(kwota_rekompensaty) as suma_wyplat,
count(case when stan_wniosku = 'wyplacony' then id
 when stan_wniosku = 'zamkniety' and kwota_rekompensaty >0 then id
  when stan_wniosku = 'wygrany w sadzie' then id
  end) as liczba_wniosków
from wnioski
where stan_wniosku like 'wyp%' or stan_wniosku like 'zamkn%' and kwota_rekompensaty >0 or stan_wniosku like 'wygrany%'
group by extract(year from data_utworzenia)


kwartal
select extract(year from data_utworzenia)as lata,date_part('quarter',data_utworzenia)as kwartal,
avg(kwota_rekompensaty)as srednia,sum(kwota_rekompensaty) as suma_wyplat,
count(case when stan_wniosku = 'wyplacony' then id
 when stan_wniosku = 'zamkniety' and kwota_rekompensaty >0 then id
  when stan_wniosku = 'wygrany w sadzie' then id
  end) as liczba_wniosków
from wnioski
where stan_wniosku like 'wyp%' or stan_wniosku like 'zamkn%' and kwota_rekompensaty >0 or stan_wniosku like 'wygrany%'
group by extract(year from data_utworzenia),date_part('quarter',data_utworzenia)

miesiac

select extract(year from data_utworzenia)as lata,extract(month from data_utworzenia)as miesiac,
avg(kwota_rekompensaty)as srednia,sum(kwota_rekompensaty) as suma_wyplat,
count(case when stan_wniosku = 'wyplacony' then id
 when stan_wniosku = 'zamkniety' and kwota_rekompensaty >0 then id
  when stan_wniosku = 'wygrany w sadzie' then id
  end) as liczba_wniosków
from wnioski
where stan_wniosku like 'wyp%' or stan_wniosku like 'zamkn%' and kwota_rekompensaty >0 or stan_wniosku like 'wygrany%'
group by extract(year from data_utworzenia),extract(month from data_utworzenia)
