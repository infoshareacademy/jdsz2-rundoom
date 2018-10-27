select count(1) as ilosc_wnioskow, extract(month from data_utworzenia),extract(year from data_utworzenia),
count(case when stan_wniosku = 'odrzucony prawnie' then id
 when stan_wniosku = 'odrzucony po analizie' then id
 when stan_wniosku = 'zamkniety' and kwota_rekompensaty = 0 then id
 when stan_wniosku = 'przegrany w sadzie' then id
 when stan_wniosku = 'odrzucony przez operatora' then id
 end)as odrzucone,
 count(case when stan_wniosku = 'odrzucony prawnie' then id
 when stan_wniosku = 'odrzucony po analizie' then id
 when stan_wniosku = 'zamkniety' and kwota_rekompensaty = 0 then id
 when stan_wniosku = 'przegrany w sadzie' then id
 when stan_wniosku = 'odrzucony przez operatora' then id
 end)/count(1)::numeric as procent_odrzuconych,
count(case when stan_wniosku = 'zaakceptowany przez operatora' then id
 when stan_wniosku = 'akcja sadowa' then id
  when stan_wniosku = 'nowy' then id
  when stan_wniosku = 'analiza zaakceptowana' then id
  when stan_wniosku = 'wysłany do operatora' then id
  end) as przetwarzany,
  count(case when stan_wniosku = 'zaakceptowany przez operatora' then id
 when stan_wniosku = 'akcja sadowa' then id
  when stan_wniosku = 'nowy' then id
  when stan_wniosku = 'analiza zaakceptowana' then id
  when stan_wniosku = 'wysłany do operatora' then id
  end)/count(1)::numeric as procent_przetwarzanych,
  count(case when stan_wniosku = 'wyplacony' then id
 when stan_wniosku = 'zamkniety' and kwota_rekompensaty >0 then id
  when stan_wniosku = 'wygrany w sadzie' then id
  end) as wyplacony,
  count(case when stan_wniosku = 'wyplacony' then id
 when stan_wniosku = 'zamkniety' and kwota_rekompensaty >0 then id
  when stan_wniosku = 'wygrany w sadzie' then id
  end)/count(1)::numeric as procent_wyplaconych


  from wnioski
group by extract(month from data_utworzenia),extract(year from data_utworzenia)
order by extract(year from data_utworzenia)


